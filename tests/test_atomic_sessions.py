from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from stacmem import StandaloneMemory
from stacmem.prepared import digest
from stacmem.standalone import IncompleteWrite
from stacmem.standalone_demo import fixture


def batch(session="s2", company="Birch", date="2024-10-15"):
    text = f"On {date} I left Aster and joined {company} as a designer."
    result = fixture("u", session, text, company, date=date)
    title = fixture("u", session, text, "designer", date=date, predicate="employment.position")
    result["cached_claims"]["claims"] += title["cached_claims"]["claims"]
    return result


def seed(memory):
    return memory.remember(**fixture("u", "s1", "On 2024-08-01 I joined Aster.", "Aster"))


def fail_at_commit(memory, monkeypatch):
    original = memory._commit_prepared

    def fail(**kwargs):
        raise RuntimeError("before batch commit")

    monkeypatch.setattr(memory, "_commit_prepared", fail)
    with pytest.raises(RuntimeError):
        memory.remember(**batch())
    monkeypatch.setattr(memory, "_commit_prepared", original)


def assert_old_state_intact(memory):
    claims = memory.memory.store.owner_claims("u")
    assert len(claims) == 1
    assert claims[0].object_value == "Aster"
    assert claims[0].status.value == "active"
    assert claims[0].valid_end is None
    assert memory.memory.store.relations_for([claims[0].id]) == []
    assert memory.memory.store.lexical_search("u", "Birch", limit=10) == []


def deny_network(*args, **kwargs):
    raise AssertionError("Recovery must not call a provider")


def test_second_claim_failure_rolls_back_old_mutations_relations_and_fts(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        seed(memory)
        original = memory.memory.conflict.ingest
        count = 0

        def fail_second(draft, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("second claim")
            return original(draft, **kwargs)

        monkeypatch.setattr(memory.memory.conflict, "ingest", fail_second)
        with pytest.raises(RuntimeError, match="second claim"):
            memory.remember(**batch())
        assert_old_state_intact(memory)
        report = memory.inspect_session(owner_id="u", session_id="s2")
        assert report["prepared_recovery_candidate"]
        assert report["recommended_action"] == "explicit_recover_prepared"
        assert not report["retry_empty_eligible"]
        with pytest.raises(IncompleteWrite, match="recover-prepared"):
            memory.retry_failed_empty(owner_id="u", session_id="s2")


def test_receipt_failure_rolls_back_entire_batch_and_recovery_attempt_is_logged(tmp_path):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        seed(memory)
        with memory._db:
            memory._db.execute("""
                CREATE TRIGGER fail_receipt BEFORE UPDATE OF status ON source_sessions
                WHEN NEW.status='committed' AND NEW.session_id='s2'
                BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END
            """)
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            memory.remember(**batch())
        assert_old_state_intact(memory)
        with pytest.raises(sqlite3.IntegrityError):
            memory.recover_prepared(owner_id="u", session_id="s2")
        assert_old_state_intact(memory)
        assert memory.status()["recovery"]["attempts"] == 1
        with memory._db:
            memory._db.execute("DROP TRIGGER fail_receipt")
        result = memory.recover_prepared(owner_id="u", session_id="s2")
        assert len(result["claims"]) == 2 and result["status"] == "committed"
        assert memory.status()["recovery"]["attempts"] == 2


def test_restart_recovery_reuses_prepared_vectors_and_replays_response(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as memory:
        seed(memory)
        fail_at_commit(memory, monkeypatch)
        saved = memory._db.execute(
            "SELECT payload_json FROM prepared_batches WHERE session_id='s2'"
        ).fetchone()[0]
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    with StandaloneMemory.offline(path) as memory:
        monkeypatch.setattr(memory.memory, "embed_drafts", deny_network)
        result = memory.recover_prepared(owner_id="u", session_id="s2")
        replay = memory.recover_prepared(owner_id="u", session_id="s2")
        assert replay == {**result, "replayed": True}
        assert memory.remember(**batch()) == replay
        assert memory.status()["messages"] == 2
        assert memory.status()["recovery"]["attempts"] == 1
        actual = memory.memory.store.get_claims(c["id"] for c in result["claims"])
        assert [c.vector for c in actual] == json.loads(saved)["vectors"]


@pytest.mark.parametrize("field", ["checksum", "request", "policy", "owner", "vectors"])
def test_tampered_or_incompatible_prepared_batch_is_blocked(tmp_path, monkeypatch, field):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        seed(memory)
        fail_at_commit(memory, monkeypatch)
        with memory._db:
            if field == "request":
                memory._db.execute(
                    "UPDATE source_sessions SET request_json='{}' WHERE session_id='s2'"
                )
            else:
                row = memory._db.execute(
                    "SELECT payload_json FROM prepared_batches WHERE session_id='s2'"
                ).fetchone()
                data = json.loads(row[0])
                if field == "policy":
                    data["policy_fingerprint"] = "different"
                elif field == "owner":
                    data["drafts"][0]["owner_id"] = "other"
                elif field == "vectors":
                    data["vectors"][0][0] = float("nan")
                else:
                    data["protocol"] = "corrupted"
                changed = json.dumps(data)
                memory._db.execute(
                    "UPDATE prepared_batches SET payload_json=?,payload_sha256=? "
                    "WHERE session_id='s2'",
                    (changed, digest(changed) if field != "checksum" else "wrong"),
                )
        with pytest.raises(ValueError):
            memory.recover_prepared(owner_id="u", session_id="s2")
        assert_old_state_intact(memory)
        assert memory.status()["recovery"]["attempts"] == 0


def test_two_instances_recover_exactly_one_batch(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as first, StandaloneMemory.offline(path) as second:
        seed(first)
        fail_at_commit(first, monkeypatch)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda memory: memory.recover_prepared(owner_id="u", session_id="s2"),
                [first, second],
            ))
        assert sum(not r["replayed"] for r in results) == 1
        assert results[0]["claims"] == results[1]["claims"]
        assert len(first.memory.store.owner_claims("u")) == 3


def test_original_writer_resumes_after_another_instance_recovers(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as first, StandaloneMemory.offline(path) as second:
        original = first._commit_prepared

        def recover_first(**kwargs):
            result = second.recover_prepared(**kwargs)
            assert not result["replayed"]
            return original(**kwargs)

        monkeypatch.setattr(first, "_commit_prepared", recover_first)
        assert first.remember(**batch())["replayed"]
        assert len(first.memory.store.owner_claims("u")) == 2


def test_exception_after_commit_does_not_demote_receipt(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        original = memory._commit_prepared

        def lose_response(**kwargs):
            original(**kwargs)
            raise RuntimeError("response lost")

        monkeypatch.setattr(memory, "_commit_prepared", lose_response)
        with pytest.raises(RuntimeError, match="response lost"):
            memory.remember(**batch())
        assert memory.status()["sessions"] == {"committed": 1}
        assert memory.remember(**batch())["replayed"]
        assert len(memory.memory.store.owner_claims("u")) == 2


@pytest.mark.parametrize("phase", ["before_prepare", "during_claims", "after_commit"])
def test_process_death_and_restart(tmp_path, phase, monkeypatch):
    path = tmp_path / "state.db"
    script = r'''
import os, sys
from stacmem import StandaloneMemory
from stacmem.standalone_demo import fixture
memory = StandaloneMemory.offline(sys.argv[1])
memory.remember(**fixture("u", "s1", "On 2024-08-01 I joined Aster.", "Aster"))
phase = sys.argv[2]
if phase == "before_prepare":
    memory.memory.embed_drafts = lambda drafts: os._exit(73)
elif phase == "during_claims":
    original = memory.memory.conflict.ingest
    def exit_in_transaction(draft, **kwargs):
        original(draft, **kwargs)
        os._exit(73)
    memory.memory.conflict.ingest = exit_in_transaction
else:
    original = memory._commit_prepared
    def exit_after_commit(**kwargs):
        original(**kwargs)
        os._exit(73)
    memory._commit_prepared = exit_after_commit
memory.remember(**fixture("u", "s2", "On 2024-10-15 I left Aster and joined Birch.",
                          "Birch", date="2024-10-15"))
'''
    process = subprocess.run(
        [sys.executable, "-c", script, str(path), phase], capture_output=True, timeout=30
    )
    assert process.returncode == 73, process.stderr.decode(errors="replace")
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    with StandaloneMemory.offline(path) as memory:
        if phase == "before_prepare":
            assert_old_state_intact(memory)
            with pytest.raises(IncompleteWrite, match="processing"):
                memory.recover_prepared(owner_id="u", session_id="s2")
        elif phase == "during_claims":
            assert_old_state_intact(memory)
            report = memory.inspect_session(owner_id="u", session_id="s2")
            assert report["receipt"]["status"] == "prepared"
            assert not memory.recover_prepared(owner_id="u", session_id="s2")["replayed"]
        else:
            assert memory.recover_prepared(owner_id="u", session_id="s2")["replayed"]
        assert memory._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_empty_prepared_batch_can_be_recovered(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        original = memory._commit_prepared
        monkeypatch.setattr(memory, "_commit_prepared", deny_network)
        data = batch()
        data["cached_claims"] = {"claims": []}
        with pytest.raises(AssertionError):
            memory.remember(**data)
        monkeypatch.setattr(memory, "_commit_prepared", original)
        assert memory.recover_prepared(owner_id="u", session_id="s2")["claims"] == []


def test_recovery_never_crosses_owner_boundary(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_at_commit(memory, monkeypatch)
        with pytest.raises(ValueError, match="No saved receipt"):
            memory.recover_prepared(owner_id="other", session_id="s2")
        assert memory.status()["recovery"]["attempts"] == 0


def test_relation_insert_failure_rolls_back_new_and_old_claims(tmp_path):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        seed(memory)
        with memory._db:
            memory._db.execute("""
                CREATE TRIGGER fail_relation BEFORE INSERT ON conflict_relations
                BEGIN SELECT RAISE(ABORT, 'injected relation failure'); END
            """)
        with pytest.raises(sqlite3.IntegrityError, match="relation failure"):
            memory.remember(**batch())
        assert_old_state_intact(memory)
        assert memory.inspect_session(owner_id="u", session_id="s2")["prepared_recovery_candidate"]


def test_recovery_cli_needs_no_credentials_even_with_online_config(tmp_path, monkeypatch, capsys):
    from stacmem.standalone_cli import main

    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as memory:
        fail_at_commit(memory, monkeypatch)
    config = tmp_path / "online.toml"
    config.write_text(
        '[extraction]\nprovider="openai_compatible"\napi_key_env="MISSING_TEST_KEY"\n'
        '[embedding]\nprovider="qwen"\n', encoding="utf-8",
    )
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(sys, "argv", [
        "stacmem", "--config", str(config), "--database", str(path),
        "recover-prepared", "--owner", "u", "--session", "s2",
    ])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "committed"


def test_legacy_partial_is_not_force_recovered(tmp_path, monkeypatch):
    from stacmem.contracts import compile_claim_payload

    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_at_commit(memory, monkeypatch)
        data = batch()
        drafts = compile_claim_payload(
            data["cached_claims"], owner_id="u", session_id="s2", messages=data["messages"]
        )
        memory.memory.ingest_drafts(drafts[:1])
        with pytest.raises(IncompleteWrite, match="Partial claims"):
            memory.recover_prepared(owner_id="u", session_id="s2")
        assert memory.memory.store.stats()["counts"]["claims"] == 1
        assert memory.status()["recovery"]["attempts"] == 0
