from __future__ import annotations

import json

import pytest

from stacmem.config import AppConfig
from stacmem.contracts import compile_claim_payload
from stacmem.models import QueryFrame
from stacmem.standalone import IncompleteWrite, ReceiptConflict, StandaloneMemory
from stacmem.standalone_demo import fixture


def payload():
    return fixture("u", "s", "On 2024-08-01 I joined Northwind.", "Northwind")


def fail_empty(memory, monkeypatch):
    original = memory.memory.embed_drafts

    def fail(drafts):
        raise RuntimeError("simulated provider failure before claim writes")

    monkeypatch.setattr(memory.memory, "embed_drafts", fail)
    with pytest.raises(RuntimeError):
        memory.remember(**payload())
    monkeypatch.setattr(memory.memory, "embed_drafts", original)


def test_recovery_after_restart_preserves_sources_and_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as memory:
        fail_empty(memory, monkeypatch)
    with StandaloneMemory.offline(path) as memory:
        result = memory.retry_failed_empty(owner_id="u", session_id="s")
        assert result["status"] == "committed" and not result["replayed"]
        assert memory.status()["messages"] == 1
        assert memory.status()["recovery"]["attempts"] == 1
        previous = json.loads(
            memory._db.execute("SELECT previous_receipt_json FROM recovery_attempts").fetchone()[0]
        )
        assert previous["status"] == "failed" and previous["error_type"] == "RuntimeError"
        assert memory.retry_failed_empty(owner_id="u", session_id="s")["replayed"]
        assert memory.status()["recovery"]["attempts"] == 1
        assert memory.memory.store.stats()["counts"]["claims"] == 1


def test_partial_write_cannot_be_recovered_as_empty(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_empty(memory, monkeypatch)
        # Reproduce an old-format partial write, outside the new atomic session path.
        data = payload()
        memory.memory.ingest_drafts(compile_claim_payload(
            data["cached_claims"], owner_id="u", session_id="s", messages=data["messages"]
        ))
        with pytest.raises(IncompleteWrite, match="Partial claims"):
            memory.retry_failed_empty(owner_id="u", session_id="s")
        assert memory.status()["recovery"]["attempts"] == 0
        assert memory.memory.store.stats()["counts"]["claims"] == 1


def test_processing_is_not_assumed_dead(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_empty(memory, monkeypatch)
        with memory._db:
            memory._db.execute("UPDATE source_sessions SET status='processing'")
        with pytest.raises(IncompleteWrite):
            memory.retry_failed_empty(owner_id="u", session_id="s")
        assert memory.status()["recovery"]["attempts"] == 0


def test_changed_input_still_rejected_with_retry_flag(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_empty(memory, monkeypatch)
        changed = payload()
        changed["messages"][0].content = "tampered"
        with pytest.raises(ReceiptConflict):
            memory.remember(**changed, retry_failed_empty=True)


def test_other_owner_cannot_recover_receipt(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_empty(memory, monkeypatch)
        with pytest.raises(ValueError, match="No saved receipt"):
            memory.retry_failed_empty(owner_id="other", session_id="s")


def test_recovery_failure_remains_blocked_and_logged(tmp_path, monkeypatch):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        fail_empty(memory, monkeypatch)

        def fail(drafts):
            raise RuntimeError("provider still unavailable")

        monkeypatch.setattr(memory.memory, "embed_drafts", fail)
        with pytest.raises(RuntimeError):
            memory.retry_failed_empty(owner_id="u", session_id="s")
        assert memory.status()["sessions"] == {"failed": 1}
        assert memory.status()["recovery"]["attempts"] == 1
        assert memory.status()["messages"] == 1


def test_recovery_handle_cannot_write_embed_or_resolve_state(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as memory:
        memory.remember(**payload())

    config = AppConfig()
    config.runtime.database_path = str(path)
    with StandaloneMemory.recovery(config) as recovery:
        assert recovery.status()["sessions"] == {"committed": 1}
        assert recovery.source_search(owner_id="u", query="Northwind")
        with pytest.raises(RuntimeError, match="recovery-only"):
            recovery.remember(**payload())
        with pytest.raises(RuntimeError, match="recovery-only"):
            recovery.search(
                owner_id="u",
                query="Where do I work?",
                frame=QueryFrame(raw_query="Where do I work?", owner_id="u"),
            )
        with pytest.raises(RuntimeError, match="recovery-only"):
            recovery.retry_failed_empty(owner_id="u", session_id="s")
        with pytest.raises(RuntimeError, match="recovery-only"):
            recovery.ask(owner_id="u", query="Where do I work?", backend=object())
