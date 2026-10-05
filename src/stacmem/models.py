"""Domain models for claims, conflict edges, queries, and evidence."""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .time_utils import now_ms


class ClaimStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"
    RETRACTED = "retracted"
    DISPUTED = "disputed"
    QUARANTINED = "quarantined"


class AdmissionDecision(StrEnum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine"


class UpdateKind(StrEnum):
    ASSERTION = "assertion"
    TRANSITION = "transition"
    CORRECTION = "correction"
    RETRACTION = "retraction"


class RelationType(StrEnum):
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    CORRECTS = "corrects"
    CONTRADICTS = "contradicts"
    COEXISTS = "coexists"
    RETRACTS = "retracts"


class TemporalIntent(StrEnum):
    CURRENT = "current"
    AS_OF = "as_of"
    HISTORY = "history"
    INTERVAL = "interval"
    UNSPECIFIED = "unspecified"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sender_id: str
    role: Literal["user", "assistant", "tool"]
    timestamp: int
    content: str
    message_id: str | None = None
    sender_name: str | None = None

    def text(self) -> str:
        return self.content


class Place(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place_id: str | None = None
    name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    hierarchy: list[str] = Field(default_factory=list)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    radius_km: float | None = Field(default=None, ge=0)
    role: str | None = None

    def normalized_key(self) -> str:
        if self.place_id:
            return self.place_id.strip().casefold()
        if self.hierarchy:
            return "/".join(part.strip().casefold() for part in self.hierarchy)
        return (self.name or "").strip().casefold()


class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_id: str
    subject: str
    predicate: str
    object_value: str
    assertion_time: int
    observed_at: int | None = None
    valid_start: int | None = None
    valid_end: int | None = None
    place: Place | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    update_kind: UpdateKind = UpdateKind.ASSERTION
    functional: bool | None = None
    source_session_id: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    source_content: str = ""
    extractor: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interval(self) -> ClaimDraft:
        if (
            self.valid_start is not None
            and self.valid_end is not None
            and self.valid_start >= self.valid_end
        ):
            raise ValueError("valid_start must be earlier than valid_end")
        return self

    @property
    def slot_key(self) -> str:
        raw = f"{normalize_text(self.subject)}|{normalize_text(self.predicate)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class Claim(ClaimDraft):
    id: str
    slot_key_value: str
    object_norm: str
    transaction_start: int
    transaction_end: int | None = None
    status: ClaimStatus = ClaimStatus.ACTIVE
    version: int = 1
    vector: list[float] | None = Field(default=None, exclude=True)

    @classmethod
    def from_draft(
        cls,
        draft: ClaimDraft,
        *,
        transaction_start: int | None = None,
        version: int = 1,
        vector: list[float] | None = None,
    ) -> Claim:
        payload = draft.model_dump()
        return cls(
            **payload,
            id=uuid.uuid4().hex,
            slot_key_value=draft.slot_key,
            object_norm=normalize_text(draft.object_value),
            transaction_start=(
                transaction_start
                if transaction_start is not None
                else (draft.observed_at if draft.observed_at is not None else now_ms())
            ),
            version=version,
            vector=vector,
        )

    @property
    def slot_key(self) -> str:
        return self.slot_key_value

    def embedding_text(self) -> str:
        place = self.place.name if self.place and self.place.name else ""
        return " ".join(
            part for part in [self.subject, self.predicate, self.object_value, place] if part
        )


class ConflictRelation(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_claim_id: str
    target_claim_id: str
    relation: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    detector: str
    created_at: int = Field(default_factory=now_ms)


class QueryFrame(BaseModel):
    raw_query: str
    owner_id: str
    target_subject: str | None = None
    target_predicate: str | None = None
    temporal_intent: TemporalIntent = TemporalIntent.UNSPECIFIED
    query_time: int | None = None
    knowledge_time: int | None = None
    interval_start: int | None = None
    interval_end: int | None = None
    place: Place | None = None
    expected_cardinality: Literal["one", "many"] = "one"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    semantic: float = 0.0
    lexical: float = 0.0
    temporal: float = 0.0
    spatial: float = 0.0
    validity: float = 0.0
    confidence: float = 0.0
    conflict_penalty: float = 0.0
    total: float = 0.0


class ScoredClaim(BaseModel):
    claim: Claim
    score: ScoreBreakdown
    decision: str = "candidate"
    suppressed_by: str | None = None


class EvidencePack(BaseModel):
    query: QueryFrame
    claims: list[ScoredClaim]
    warnings: list[str] = Field(default_factory=list)
    context: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CommitWatermark(BaseModel):
    commit_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str
    owner_id: str
    app_id: str
    project_id: str
    received_at: int = Field(default_factory=now_ms)
    extraction_completed_at: int | None = None
    ledger_committed_at: int | None = None
    search_visible_at: int | None = None
    status: str = "received"
    details: dict[str, Any] = Field(default_factory=dict)


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
