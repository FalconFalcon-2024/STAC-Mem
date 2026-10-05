"""Protocols for model-dependent compilation steps."""

from __future__ import annotations

from typing import Protocol

from stacmem.models import ClaimDraft, Message, QueryFrame


class ClaimExtractor(Protocol):
    name: str

    def extract(
        self,
        *,
        owner_id: str,
        session_id: str,
        messages: list[Message],
    ) -> list[ClaimDraft]: ...


class QueryCompiler(Protocol):
    name: str

    def compile(self, *, owner_id: str, query: str) -> QueryFrame: ...
