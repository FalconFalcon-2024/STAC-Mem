import pytest
from pydantic import ValidationError

from stacmem import StandaloneMemory
from stacmem.contracts import compile_claim_payload
from stacmem.inspection import inspect_session
from stacmem.standalone_api import RememberRequest
from stacmem.standalone_demo import fixture


@pytest.mark.parametrize("partial", [False, True])
def test_failed_inspection_is_read_only_and_redacts_source(tmp_path, monkeypatch, partial):
    path = tmp_path / "state.db"
    with StandaloneMemory.offline(path) as memory:
        original = memory.memory.embed_drafts

        def fail(drafts):
            raise RuntimeError("failure")

        monkeypatch.setattr(memory.memory, "embed_drafts", fail)
        data = fixture("u", "s", "On 2024-08-01 I joined Northwind.", "Northwind")
        with pytest.raises(RuntimeError):
            memory.remember(**data)
        monkeypatch.setattr(memory.memory, "embed_drafts", original)
        if partial:
            memory.memory.ingest_drafts(compile_claim_payload(
                data["cached_claims"], owner_id="u", session_id="s", messages=data["messages"]
            ))
        before = memory.status()
        result = memory.inspect_session(owner_id="u", session_id="s")
        assert result["retry_empty_eligible"] is not partial
        assert result["claim_count"] == int(partial)
        assert result["network_calls"] == 0
        assert "Northwind" not in str(result)
        assert memory.status() == before
        with pytest.raises(ValueError, match="No saved receipt"):
            inspect_session(path, owner_id="other", session_id="s")


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        inspect_session(path, owner_id="u", session_id="s")
    assert not path.exists()


def test_processing_is_not_declared_recoverable(tmp_path):
    with StandaloneMemory.offline(tmp_path / "state.db") as memory:
        memory.remember(**fixture("u", "s", "On 2024-08-01 I joined Northwind.", "Northwind"))
        with memory._db:
            memory._db.execute("UPDATE source_sessions SET status='processing'")
        result = memory.inspect_session(owner_id="u", session_id="s")
        assert not result["retry_empty_eligible"]
        assert result["recommended_action"] == "check_writer_liveness_do_not_replay"


@pytest.mark.parametrize("content", [[], [{"type": "text", "text": "hello"}], "   "])
def test_api_declares_text_only_contract(content):
    with pytest.raises(ValidationError):
        RememberRequest(owner_id="u", session_id="s", messages=[
            {"sender_id": "u", "role": "user", "timestamp": 1, "content": content}
        ])
