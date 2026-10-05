from __future__ import annotations

import json
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from stacmem.models import QueryFrame
from stacmem.standalone import IncompleteWrite, ReceiptConflict, StandaloneMemory
from stacmem.standalone_demo import fixture, run


@pytest.fixture
def standalone(tmp_path):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        yield memory


def payload():
    return fixture("user", "session", "On 2024-08-01 I joined Northwind.", "Northwind")


def test_demo_has_no_network_and_passes(tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Offline demo attempted network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    result = run(tmp_path / "demo")
    assert result["passed"] == result["checks"] == 10
    assert result["status"]["sessions"] == {"committed": 6}
    records = [
        json.loads(line)
        for line in (tmp_path / "demo" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 16
    assert [r["check"] for r in records if "check" in r] == result["results"]
    assert next(r for r in records if r["kind"] == "retry_check")["result"]["replayed"]
    assert len(next(r for r in records if r["kind"] == "source_check")["results"]) == 2
    assert next(r for r in records if r["kind"] == "isolation_check")["results"] == []
    with pytest.raises(FileExistsError):
        run(tmp_path / "demo")


def test_identical_retry_does_not_duplicate_after_restart(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as memory:
        original = memory.remember(**payload())
    with StandaloneMemory.offline(path) as memory:
        replay = memory.remember(**payload())
        assert replay["replayed"] is True
        assert replay["claims"] == original["claims"]
        assert memory.status()["messages"] == 1
        assert memory.memory.store.stats()["counts"]["claims"] == 1


def test_reusing_session_with_changed_payload_rejected(standalone):
    standalone.remember(**payload())
    changed = payload()
    changed["messages"][0].content = "different text"
    with pytest.raises(ReceiptConflict):
        standalone.remember(**changed)


def test_same_session_other_owner_is_independent(standalone):
    standalone.remember(**payload())
    other = fixture("other", "session", "On 2024-08-01 I joined Contoso.", "Contoso")
    standalone.remember(**other)
    assert standalone.source_search(owner_id="other", query="Northwind") == []
    assert len(standalone.source_search(owner_id="user", query="Northwind")) == 1


def test_failure_preserves_source_and_blocks_owner(standalone, monkeypatch):
    def fail(*args):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(standalone.memory, "embed_drafts", fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        standalone.remember(**payload())
    assert standalone.status()["sessions"] == {"failed": 1}
    assert (
        standalone.source_search(owner_id="user", query="Northwind")[0]["extraction_status"]
        == "failed"
    )
    with pytest.raises(IncompleteWrite):
        standalone.remember(**payload())
    with pytest.raises(IncompleteWrite):
        standalone.remember(**fixture("user", "new", "I joined Example.", "Example"))
    frame = QueryFrame(raw_query="employer", owner_id="user")
    with pytest.raises(IncompleteWrite):
        standalone.search(owner_id="user", query="employer", frame=frame)


def test_query_frame_cannot_cross_owner(standalone):
    with pytest.raises(ValueError, match="must match"):
        standalone.search(
            owner_id="other", query="q", frame=QueryFrame(raw_query="q", owner_id="user")
        )


def test_empty_extraction_still_preserves_original(standalone):
    data = payload()
    data["cached_claims"] = {"claims": []}
    assert standalone.remember(**data)["claims"] == []
    assert len(standalone.source_search(owner_id="user", query="Northwind")) == 1


def test_invalid_source_quarantined_not_lost(standalone):
    data = payload()
    data["cached_claims"]["claims"][0]["object_surface"] = "Invented"
    assert standalone.remember(**data)["claims"][0]["status"] == "quarantined"
    assert standalone.source_search(owner_id="user", query="Northwind")


def test_concurrent_identical_requests_are_idempotent(standalone):
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: standalone.remember(**payload()), range(3)))
    assert sum(not r["replayed"] for r in results) == 1
    assert standalone.status()["messages"] == 1


def test_legacy_database_is_never_upgraded(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE claims (id TEXT)")
    with pytest.raises(ValueError, match="new database"):
        StandaloneMemory.offline(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("claims",)
        ]


def test_duplicate_message_ids_rejected_before_write(standalone):
    data = payload()
    data["messages"] *= 2
    with pytest.raises(ValueError, match="unique"):
        standalone.remember(**data)
    assert standalone.status()["messages"] == 0


def test_offline_natural_input_is_not_silently_treated_as_extracted(standalone):
    data = payload()
    data.pop("cached_claims")
    with pytest.raises(ValueError, match="cached_claims"):
        standalone.remember(**data)
    assert standalone.status()["messages"] == 0


def test_api_contract_and_errors(standalone):
    from fastapi.testclient import TestClient

    from stacmem.standalone_api import create_app

    data = payload()
    data["messages"] = [m.model_dump(mode="json") for m in data["messages"]]
    with TestClient(create_app(standalone)) as client:
        assert client.get("/health").json()["profile"] == "standalone-state-v1"
        assert client.post("/api/v1/session/remember", json=data).status_code == 422
    with TestClient(create_app(standalone, allow_cached_claims=True)) as client:
        assert client.post("/api/v1/session/remember", json=data).status_code == 200
        assert client.post("/api/v1/session/remember", json=data).json()["replayed"] is True
        data["messages"][0]["content"] = "changed"
        assert client.post("/api/v1/session/remember", json=data).status_code == 409
        assert (
            client.post(
                "/api/v1/memory/search",
                json={
                    "owner_id": "other",
                    "query": "q",
                    "frame": {"raw_query": "q", "owner_id": "user"},
                },
            ).status_code
            == 422
        )
        assert client.post(
            "/api/v1/sources/search", json={"owner_id": "user", "query": "Northwind"}
        ).json()["sources"]


def test_failed_atomic_batch_never_silently_retried(standalone, monkeypatch):
    original = standalone.memory.ingest_prepared

    def fail_after_claims(drafts, vectors):
        original(drafts, vectors)
        raise RuntimeError("crash before receipt commit")

    monkeypatch.setattr(standalone.memory, "ingest_prepared", fail_after_claims)
    with pytest.raises(RuntimeError):
        standalone.remember(**payload())
    assert standalone.memory.store.stats()["counts"]["claims"] == 0
    with pytest.raises(IncompleteWrite):
        standalone.remember(**payload())
    assert standalone.memory.store.stats()["counts"]["claims"] == 0


def test_other_instance_cannot_write_same_owner_during_extraction(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as first, StandaloneMemory.offline(path) as second:
        original = first.memory.embed_drafts

        def try_overlap(drafts):
            with pytest.raises(IncompleteWrite):
                second.remember(**fixture("user", "other", "I joined Example.", "Example"))
            return original(drafts)

        monkeypatch.setattr(first.memory, "embed_drafts", try_overlap)
        first.remember(**payload())
        assert second.remember(**payload())["replayed"] is True


def test_owner_changed_during_read_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as first, StandaloneMemory.offline(path) as second:
        first.remember(**payload())
        original = first.memory.search

        def write_during_read(**kwargs):
            result = original(**kwargs)
            second.remember(**fixture("user", "later", "I joined Contoso.", "Contoso"))
            return result

        monkeypatch.setattr(first.memory, "search", write_during_read)
        with pytest.raises(IncompleteWrite, match="changed during search"):
            first.search(
                owner_id="user", query="q", frame=QueryFrame(
                    raw_query="q", owner_id="user", target_predicate="employment.organization"
                )
            )
