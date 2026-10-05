from __future__ import annotations

import hashlib
import json
import socket
import sqlite3

import pytest

from stacmem.admission import AdmissionAssessment
from stacmem.config import AppConfig
from stacmem.embeddings import HashingEmbedder
from stacmem.migration import MigrationBlocked, revalidate_history
from stacmem.models import AdmissionDecision, QueryFrame
from stacmem.pipeline import StacMemory
from stacmem.prepared import digest
from stacmem.standalone import IncompleteWrite, StandaloneMemory
from stacmem.standalone_cli import main
from stacmem.standalone_demo import fixture
from stacmem.time_utils import ensure_ms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deny_network(*args, **kwargs):
    raise AssertionError("Offline revalidation attempted network")


def seed(memory):
    memory.remember(**fixture("u", "initial", "On 2024-08-01 I joined Northwind.", "Northwind"))


def unsafe_old_admission(draft, **kwargs):
    # Simulate the superseded policy trusting positive model interpretation.
    draft.metadata["proposition_certificate"] = {"transition_entailment": True}
    return AdmissionAssessment(AdmissionDecision.ACCEPT)


def test_audit_and_rebuild_repair_old_admission_and_restore_superseded_state(tmp_path, monkeypatch):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
        monkeypatch.setattr(memory.memory.conflict.admission_gate, "assess", unsafe_old_admission)
        data = fixture("u", "later", "In a hypothetical example, I work at Aster.", "Aster",
                       date="2024-10-01")
        data["cached_claims"]["claims"][0]["source_content"] = "I work at Aster."
        old = memory.remember(**data)
        assert old["claims"][0]["status"] == "active"
        prior = next(c for c in memory.memory.store.owner_claims("u")
                     if c.object_value == "Northwind")
        assert prior.status == "superseded"
    before = sha(source)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    audit = revalidate_history(source)
    assert audit["changed_claims"] == 2
    assert audit["claim_statuses"] == {"active": 1, "quarantined": 1}
    assert audit["relation_counts_before"] == {"supersedes": 1}
    assert audit["relation_counts_after"] == {}
    assert len(audit["relation_changes"]["removed"]) == 1
    assert audit["status"] == "completed" and audit["extraction_calls"] == 0
    assert sha(source) == before
    output = tmp_path / "rebuilt"
    rebuilt = revalidate_history(source, output=output)
    assert rebuilt["changed_claims"] == audit["changed_claims"]
    assert not list(tmp_path.glob(".rebuilt.staging-*"))
    assert sha(source) == before
    with StandaloneMemory.offline(output / "memory.sqlite3") as memory:
        question = "Where do I work?"
        frame = QueryFrame(raw_query=question, owner_id="u", target_subject="u",
                           target_predicate="employment.organization", temporal_intent="current",
                           query_time=ensure_ms("2025-01-01"))
        result = memory.search(owner_id="u", query=question, frame=frame)
        assert [c["claim"]["object_value"] for c in result["claims"]] == ["Northwind"]
        assert memory.status()["messages"] == 2
    with sqlite3.connect(output / "source_snapshot.sqlite3") as db:
        assert db.execute("SELECT status FROM claims WHERE object_value='Aster'").fetchone() == (
            "active",
        )
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_relation_type_change_is_reported_without_claim_status_change(tmp_path):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
        memory.remember(**fixture("u", "later", "Since 2024-10-01 I joined Aster.",
                                  "Aster", date="2024-10-01"))
    with sqlite3.connect(source) as db:
        assert db.execute("SELECT relation FROM conflict_relations").fetchone() == (
            "supersedes",
        )
        db.execute("UPDATE conflict_relations SET relation='coexists'")

    report = revalidate_history(source)

    assert report["changed_claims"] == 0
    assert report["relation_counts_before"] == {"coexists": 1}
    assert report["relation_counts_after"] == {"supersedes": 1}
    assert len(report["relation_changes"]["changed"]) == 1


def test_old_prepared_policy_can_be_revalidated_without_old_runtime(tmp_path, monkeypatch):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
        original = memory._commit_prepared

        def stop(**kwargs):
            raise RuntimeError("crash before commit")

        monkeypatch.setattr(memory, "_commit_prepared", stop)
        with pytest.raises(RuntimeError):
            memory.remember(**fixture("u", "pending", "My friend works at Aster.", "Aster"))
        monkeypatch.setattr(memory, "_commit_prepared", original)
        with memory._db:
            row = memory._db.execute("SELECT payload_json FROM prepared_batches").fetchone()
            payload = json.loads(row[0])
            payload["policy_fingerprint"] = "previous-grounding-policy"
            encoded = json.dumps(payload)
            memory._db.execute("UPDATE prepared_batches SET payload_json=?,payload_sha256=?",
                               (encoded, digest(encoded)))
        with pytest.raises(ValueError, match="policy_fingerprint"):
            memory.recover_prepared(owner_id="u", session_id="pending")
    before = sha(source)
    assert revalidate_history(source)["status"] == "blocked"
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    result = revalidate_history(source, output=tmp_path / "new", include_prepared=True)
    assert result["prepared_sessions_revalidated"] == 1
    assert result["previous_prepared_policies"] == ["previous-grounding-policy"]
    assert result["claim_statuses"] == {"active": 1, "quarantined": 1}
    assert sha(source) == before
    with sqlite3.connect(source) as db:
        assert db.execute(
            "SELECT status FROM source_sessions WHERE session_id='pending'"
        ).fetchone() == ("failed",)
        assert db.execute("SELECT count(*) FROM prepared_batches").fetchone()[0] == 1


def test_incomplete_source_without_saved_candidates_blocks_output(tmp_path, monkeypatch):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        def stop(*args):
            raise RuntimeError("failed before preparing")

        monkeypatch.setattr(memory.memory, "embed_drafts", stop)
        with pytest.raises(RuntimeError):
            seed(memory)
    output = tmp_path / "new"
    with pytest.raises(MigrationBlocked) as failure:
        revalidate_history(source, output=output, include_prepared=True)
    assert failure.value.report["blocked_sessions"][0]["reason"] == (
        "no_saved_candidates_requires_new_extraction"
    )
    assert not output.exists()


def test_existing_output_and_corrupt_request_are_not_replayed(tmp_path):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
    with pytest.raises(FileExistsError):
        revalidate_history(source, output=tmp_path)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE source_sessions SET request_json='{}'")
    with pytest.raises(ValueError, match="checksum"):
        revalidate_history(source, output=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_partial_rebuild_blocks_both_public_entry_points(tmp_path, monkeypatch):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)

    def stop(*args):
        raise RuntimeError("embedding failed during rebuild")

    monkeypatch.setattr(HashingEmbedder, "embed", stop)
    output = tmp_path / "new"
    with pytest.raises(RuntimeError):
        revalidate_history(source, output=output)
    assert not output.exists()
    stages = list(tmp_path.glob(".new.staging-*"))
    assert len(stages) == 1
    stage = stages[0]
    assert json.loads((stage / "report.json").read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(IncompleteWrite, match="Historical rebuild"):
        StandaloneMemory.offline(stage / "memory.sqlite3")
    config = AppConfig()
    config.runtime.database_path = str(stage / "memory.sqlite3")
    with pytest.raises(ValueError, match="Historical rebuild"):
        StacMemory.from_app_config(config)


def test_provider_init_failure_leaves_only_blocked_staging_database(tmp_path, monkeypatch):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
    before = sha(source)

    def stop(*args, **kwargs):
        raise RuntimeError("provider initialization failed")

    monkeypatch.setattr("stacmem.pipeline.OpenAICompatibleEmbedder.__init__", stop)
    config = AppConfig()
    config.embedding.provider = "qwen"
    output = tmp_path / "new"
    with pytest.raises(RuntimeError, match="provider initialization failed"):
        revalidate_history(source, config=config, output=output, allow_paid_api=True)
    assert not output.exists() and sha(source) == before
    stages = list(tmp_path.glob(".new.staging-*"))
    assert len(stages) == 1
    stage = stages[0]
    report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["error_type"] == "RuntimeError"
    with pytest.raises(IncompleteWrite, match="Historical rebuild"):
        StandaloneMemory.offline(stage / "memory.sqlite3")


def test_cli_history_audit_does_not_open_online_providers(tmp_path, monkeypatch, capsys):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr("sys.argv", ["stacmem", "--offline", "--database", str(source),
                                    "audit-history"])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["claims_revalidated"] == 1 and result["changed_claims"] == 0


def test_split_source_and_ledger_schema_cannot_be_reinterpreted(tmp_path):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE standalone_meta SET value='{}' WHERE key='predicate_schema'")
    with pytest.raises(ValueError, match="predicate_schema differs"):
        revalidate_history(source, output=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_orphan_source_messages_are_not_silently_dropped(tmp_path):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        seed(memory)
    with sqlite3.connect(source) as db:
        db.execute("INSERT INTO source_messages VALUES ('u','orphan','m',1,'user','lost source')")
    with pytest.raises(ValueError, match="Orphaned"):
        revalidate_history(source, output=tmp_path / "new")


def test_rebuild_reports_evidence_floor_change_even_if_unknown_bounds_stay_unknown(tmp_path):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        data = fixture("u", "undated", "I work at Aster.", "Aster",
                       date="2024-10-01", kind="assertion")
        memory.remember(**data)
    # The previous runtime saved unknown bounds but no evidence eligibility contract.
    with sqlite3.connect(source) as db:
        claim_id, encoded = db.execute("SELECT id,metadata_json FROM claims").fetchone()
        metadata = json.loads(encoded)
        metadata.pop("temporal_eligibility")
        db.execute("UPDATE claims SET metadata_json=? WHERE id=?",
                   (json.dumps(metadata), claim_id))
    before = sha(source)
    output = tmp_path / "rebuilt"
    report = revalidate_history(source, output=output)
    assert report["changed_claims"] == 1
    change = report["changes"][0]
    assert change["before"]["valid_start"] is change["after"]["valid_start"] is None
    assert change["before"]["temporal_eligibility"] is None
    assert change["after"]["temporal_eligibility"]["kind"] == "unknown_onset"
    assert sha(source) == before
    with StandaloneMemory.offline(output / "memory.sqlite3") as memory:
        for date, expected in [("2024-09-30", []), ("2025-01-01", ["Aster"])]:
            query = QueryFrame(
                owner_id="u", raw_query="Where do I work?", target_subject="u",
                target_predicate="employment.organization", temporal_intent="as_of",
                query_time=ensure_ms(date),
            )
            pack = memory.search(owner_id="u", query=query.raw_query, frame=query)
            assert [item["claim"]["object_value"] for item in pack["claims"]] == expected


def test_rebuild_repairs_old_endpoint_attached_to_new_transition_value(tmp_path):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        memory.remember(**fixture(
            "u", "arrival", "I lived in Paris until 2025-01-01 then moved to London.",
            "London", date="2025-02-01", predicate="residence.current",
        ))
    # Simulate a committed ledger produced before the temporal-scope fix.
    with sqlite3.connect(source) as db:
        encoded = db.execute("SELECT metadata_json FROM claims").fetchone()[0]
        metadata = json.loads(encoded)
        metadata["temporal_eligibility"]["kind"] = "source_bounded"
        metadata["temporal_eligibility"]["eligible_from"] = None
        db.execute("UPDATE claims SET valid_end=?,metadata_json=?",
                   (ensure_ms("2025-01-01"), json.dumps(metadata)))
    before = sha(source)
    output = tmp_path / "rebuilt"
    report = revalidate_history(source, output=output)
    assert report["changed_claims"] == 1 and report["status"] == "completed"
    change = report["changes"][0]
    assert change["before"]["valid_end"] == ensure_ms("2025-01-01")
    assert change["after"]["valid_end"] is None
    assert sha(source) == before
    with StandaloneMemory.offline(output / "memory.sqlite3") as memory:
        for date, expected in [("2024-12-31", []), ("2025-02-01", ["London"])]:
            frame = QueryFrame(
                owner_id="u", target_subject="u", target_predicate="residence.current",
                raw_query="Where did I live?", temporal_intent="as_of", query_time=ensure_ms(date),
            )
            pack = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
            assert [c["claim"]["object_value"] for c in pack["claims"]] == expected


@pytest.mark.parametrize("modifier", [
    "by 2025-03-01", "in Q1 2025", "two months afterward",
])
def test_rebuild_clears_old_shared_onset_when_target_had_an_unparsed_date(tmp_path, modifier):
    source = tmp_path / "old.db"
    with StandaloneMemory.offline(source) as memory:
        memory.remember(**fixture(
            "u", "arrival",
            f"On 2025-01-01 I joined Aster and moved to London {modifier}.",
            "London", date="2025-04-01", predicate="residence.current",
        ))
    # Reproduce the previous committed ledger, not a new model extraction.
    with sqlite3.connect(source) as db:
        encoded = db.execute("SELECT metadata_json FROM claims").fetchone()[0]
        metadata = json.loads(encoded)
        metadata["temporal_eligibility"].update(
            kind="source_bounded", eligible_from=ensure_ms("2025-01-01"), asserts_onset=True,
        )
        db.execute("UPDATE claims SET valid_start=?,metadata_json=?",
                   (ensure_ms("2025-01-01"), json.dumps(metadata)))
    before = sha(source)
    output = tmp_path / "rebuilt"
    report = revalidate_history(source, output=output)
    assert report["status"] == "completed" and report["changed_claims"] == 1
    assert report["changes"][0]["before"]["valid_start"] == ensure_ms("2025-01-01")
    assert report["changes"][0]["after"]["valid_start"] is None
    assert report["changes"][0]["after"]["temporal_eligibility"]["kind"] == "unknown_onset"
    assert sha(source) == before
    with StandaloneMemory.offline(output / "memory.sqlite3") as memory:
        for date, expected in [("2025-01-01", []), ("2025-04-01", ["London"])]:
            frame = QueryFrame(
                owner_id="u", target_subject="u", target_predicate="residence.current",
                raw_query="Where did I live?", temporal_intent="as_of", query_time=ensure_ms(date),
            )
            pack = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
            assert [c["claim"]["object_value"] for c in pack["claims"]] == expected
