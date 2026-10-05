"""Owner-bound tools for plugging STAC-Mem into an agent function-call loop."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import Message, QueryFrame, canonical_json
from .standalone import StandaloneMemory


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecallStateInput(_ToolInput):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchSourcesInput(RecallStateInput):
    pass


class BoundAgentMemoryTools:
    """Expose read-only model tools without letting a model choose another owner.

    The host binds the authenticated owner and sends genuine transcript events through
    ``record_host_message``. Model function calls can only retrieve memory. Errors remain
    visible to the host loop.
    """

    def __init__(self, memory: StandaloneMemory, *, owner_id: str) -> None:
        if not owner_id.strip():
            raise ValueError("owner_id must not be blank")
        self.memory = memory
        self.owner_id = owner_id

    def record_host_message(
        self, *, message: Message, conversation_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Record one host event; retries reuse its conversation/message receipt ID.

        ``session_id`` is an advanced explicit ingestion ID, unique per event. It is
        retained for callers of the initial adapter; it cannot accompany conversation_id.
        """
        if message.role != "user" or message.sender_id != self.owner_id:
            raise ValueError("Host message must be from the bound authenticated user")
        if not message.message_id or not message.message_id.strip():
            raise ValueError("Host message requires a stable transcript message_id")
        if (conversation_id is None) == (session_id is None):
            raise ValueError("Supply exactly one of conversation_id or explicit session_id")
        if conversation_id is not None:
            if not conversation_id.strip():
                raise ValueError("conversation_id must not be blank")
            identity = canonical_json([conversation_id, message.message_id])
            session_id = "host-message-v1:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        elif not session_id.strip():
            raise ValueError("session_id must not be blank")
        return self.memory.remember(
            owner_id=self.owner_id,
            session_id=session_id,
            messages=[message],
        )

    def recall_state(
        self,
        *,
        query: str,
        top_k: int = 10,
        frame: QueryFrame | None = None,
    ) -> dict[str, Any]:
        data = RecallStateInput(query=query, top_k=top_k)
        return self.memory.search(
            owner_id=self.owner_id,
            query=data.query,
            top_k=data.top_k,
            frame=frame,
        )

    def search_memory_sources(self, *, query: str, top_k: int = 10) -> dict[str, Any]:
        data = SearchSourcesInput(query=query, top_k=top_k)
        return {
            "kind": "raw_sources_not_resolved_state",
            "sources": self.memory.source_search(
                owner_id=self.owner_id, query=data.query, top_k=data.top_k
            ),
        }

    def tool_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function definitions used by many agent runtimes."""
        return [
            self._spec(
                "recall_state",
                "Resolve current or historical user state. Treat warnings and empty claims as "
                "unknown, not as permission to guess.",
                RecallStateInput,
            ),
            self._spec(
                "search_memory_sources",
                "Search original messages when resolved state is unavailable. Results are raw "
                "sources and are not a current-state answer.",
                SearchSourcesInput,
            ),
        ]

    def invoke(self, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Validate and execute one tool call from an agent runtime."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        if name == "recall_state":
            data = RecallStateInput.model_validate(arguments)
            return self.recall_state(**data.model_dump())
        if name == "search_memory_sources":
            data = SearchSourcesInput.model_validate(arguments)
            return self.search_memory_sources(**data.model_dump())
        raise ValueError(f"Unknown memory tool: {name}")

    @staticmethod
    def _spec(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
                "strict": True,
            },
        }
