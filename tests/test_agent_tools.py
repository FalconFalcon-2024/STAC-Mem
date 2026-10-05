from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from stacmem.agent_tools import BoundAgentMemoryTools
from stacmem.models import Message
from stacmem.standalone import ReceiptConflict, StandaloneMemory
from stacmem.standalone_demo import fixture


class FakeMemory:
    def __init__(self):
        self.calls = []

    def remember(self, **kwargs):
        self.calls.append(("remember", kwargs))
        return {"status": "committed"}

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return {"claims": [], "warnings": ["unknown"]}

    def source_search(self, **kwargs):
        self.calls.append(("sources", kwargs))
        return [{"content": "saved source"}]


def test_message_sdk_is_text_only():
    with pytest.raises(ValidationError):
        Message(sender_id="u", role="user", timestamp=1, content=[{"text": "hello"}])


def test_agent_tools_bind_owner_and_dispatch_validated_calls():
    memory = FakeMemory()
    tools = BoundAgentMemoryTools(memory, owner_id="authenticated-user")
    event = Message(
        sender_id="authenticated-user", role="user", timestamp=1,
        content="I moved to Bern.", message_id="host-event-1",
    )
    result = tools.record_host_message(session_id="s1", message=event)
    assert result == {"status": "committed"}
    _, call = memory.calls[-1]
    assert call["owner_id"] == "authenticated-user"
    assert call["messages"] == [event]
    assert "owner_id" not in str(tools.tool_specs())
    assert "remember_memory" not in str(tools.tool_specs())

    evidence = tools.invoke("recall_state", '{"query":"Where do I live?","top_k":3}')
    assert evidence["warnings"] == ["unknown"]
    assert memory.calls[-1][1]["owner_id"] == "authenticated-user"


def test_agent_tools_reject_owner_injection_and_unknown_tools():
    tools = BoundAgentMemoryTools(FakeMemory(), owner_id="u")
    with pytest.raises(ValueError, match="Unknown memory tool"):
        tools.invoke("remember_memory", {"session_id": "s", "text": "invented user claim"})
    with pytest.raises(ValueError, match="authenticated user"):
        tools.record_host_message(session_id="s", message=Message(
            sender_id="agent", role="assistant", timestamp=1, content="The user works at Aster."
        ))
    with pytest.raises(ValueError, match="authenticated user"):
        tools.record_host_message(session_id="s", message=Message(
            sender_id="another-user", role="user", timestamp=1, content="I work at Aster."
        ))
    with pytest.raises(ValidationError):
        tools.invoke("recall_state", {"query": "q", "owner_id": "other"})
    with pytest.raises(ValueError, match="Unknown memory tool"):
        tools.invoke("delete_everything", {})


def test_assistant_source_is_audited_but_not_active_user_state(tmp_path):
    data = fixture("u", "s1", "The user works at Aster.", "Aster")
    data["messages"][0].sender_id = "agent"
    data["messages"][0].role = "assistant"
    with StandaloneMemory.offline(tmp_path / "memory.sqlite3") as memory:
        result = memory.remember(**data)
        assert result["status"] == "committed"
        assert result["claims"][0]["status"] == "quarantined"
        assert memory.source_search(owner_id="u", query="Aster")


def test_two_host_events_in_one_conversation_have_separate_receipts(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.sqlite3") as memory:
        memory.memory.claim_extractor = SimpleNamespace(extract=lambda **kwargs: [])
        tools = BoundAgentMemoryTools(memory, owner_id="u")
        first = Message(sender_id="u", role="user", timestamp=1, content="hello one",
                        message_id="m1")
        second = first.model_copy(update={"content": "hello two", "message_id": "m2"})
        a = tools.record_host_message(conversation_id="chat-1", message=first)
        b = tools.record_host_message(conversation_id="chat-1", message=second)
        assert a["status"] == b["status"] == "committed"
        assert a["session_id"] != b["session_id"]
        assert tools.record_host_message(conversation_id="chat-1", message=first)["replayed"]
        with pytest.raises(ReceiptConflict):
            tools.record_host_message(conversation_id="chat-1", message=first.model_copy(
                update={"content": "changed content under the same event"}
            ))
        assert memory.status()["messages"] == 2
        assert len(memory.source_search(owner_id="u", query="hello")) == 2


@pytest.mark.parametrize("message_id", [None, "", "  "])
def test_host_adapter_requires_real_stable_message_id(message_id):
    tools = BoundAgentMemoryTools(FakeMemory(), owner_id="u")
    event = Message(sender_id="u", role="user", timestamp=1, content="hello",
                    message_id=message_id)
    with pytest.raises(ValueError, match="message_id"):
        tools.record_host_message(conversation_id="chat-1", message=event)


def test_conversation_event_identity_is_unambiguous():
    memory = FakeMemory()
    tools = BoundAgentMemoryTools(memory, owner_id="u")
    event = Message(sender_id="u", role="user", timestamp=1, content="hello", message_id="c")
    tools.record_host_message(conversation_id="a:b", message=event)
    receipt_one = memory.calls[-1][1]["session_id"]
    tools.record_host_message(conversation_id="a", message=event.model_copy(
        update={"message_id": "b:c"}
    ))
    assert memory.calls[-1][1]["session_id"] != receipt_one
    with pytest.raises(ValueError, match="exactly one"):
        tools.record_host_message(conversation_id="chat", session_id="event", message=event)
