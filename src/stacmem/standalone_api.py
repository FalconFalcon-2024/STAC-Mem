"""Loopback-only development API. No authentication or multi-tenant security claim."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import Message, QueryFrame
from .standalone import IncompleteWrite, ReceiptConflict, StandaloneMemory


class TextMessage(Message):
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A nonempty text message is required")
        return value


class RememberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    messages: list[TextMessage] = Field(min_length=1, max_length=200)
    cached_claims: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=10000)
    frame: QueryFrame | None = None
    top_k: int = Field(default=10, ge=1, le=100)


def create_app(memory: StandaloneMemory, *, allow_cached_claims: bool = False) -> FastAPI:
    """Caller owns/ closes memory; CLI binds this app to loopback only."""
    app = FastAPI(title="STAC-Mem API", version="0.1.0")

    @app.exception_handler(ReceiptConflict)
    @app.exception_handler(IncompleteWrite)
    async def conflict_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=409, content={"error": type(exc).__name__, "detail": str(exc)}
        )

    @app.exception_handler(ValueError)
    async def validation_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422, content={"error": "InvalidRequest", "detail": str(exc)}
        )

    @app.get("/health")
    def health():
        return memory.status()

    @app.post("/api/v1/session/remember")
    def remember(data: RememberRequest):
        if data.cached_claims is not None and not allow_cached_claims:
            raise ValueError("Cached structures are disabled in this server")
        return memory.remember(**data.model_dump(exclude={"messages"}), messages=data.messages)

    @app.post("/api/v1/memory/search")
    def search(data: SearchRequest):
        return memory.search(**data.model_dump(exclude={"frame"}), frame=data.frame)

    @app.post("/api/v1/sources/search")
    def sources(data: SearchRequest):
        if data.frame is not None:
            raise ValueError(
                "Source search is lexical only; QueryFrame constraints are unsupported"
            )
        return {
            "kind": "raw_sources_not_resolved_state",
            "sources": memory.source_search(
                owner_id=data.owner_id, query=data.query, top_k=data.top_k
            ),
        }

    return app
