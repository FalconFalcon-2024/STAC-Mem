"""Integrity and compatibility checks for durable, precomputed ingestion batches."""

from __future__ import annotations

import hashlib
import json
import math

from .config import AppConfig
from .models import ClaimDraft, canonical_json

PROTOCOL = "atomic-prepared-v2"
POLICY_SEMANTICS = {
    "claim_contract": "source-full-message-v2",
    "admission": "deterministic-proposition-admission-v7",
    "conflict": "grounded-scoped-evidence-interval-v3",
    "temporal_support": "positive-temporal-scope-v8",
    "observation": "observation-contract-v2",
    "resolver": "scoped-state-evidence-floor-v2",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def policy_fingerprint(config: AppConfig) -> str:
    # Explicit semantic versions change only when replay meaning changes, not on formatting edits.
    return digest(canonical_json({
        "protocol": PROTOCOL,
        "semantics": POLICY_SEMANTICS,
        "conflict_config": config.conflict.model_dump(mode="json"),
        "admission_config": config.admission.model_dump(mode="json"),
        "temporal_grounding_config": config.temporal_grounding.model_dump(mode="json"),
        "predicate_schema": config.predicate_schema.model_dump(mode="json"),
        "place_schema": config.place_schema.model_dump(mode="json"),
    }))


def encode_batch(
    drafts,
    vectors,
    *,
    owner_id,
    session_id,
    request_fingerprint,
    policy,
    embedding_space,
) -> str:
    value = canonical_json({
        "protocol": PROTOCOL,
        "owner_id": owner_id,
        "session_id": session_id,
        "request_fingerprint": request_fingerprint,
        "policy_fingerprint": policy,
        "embedding_space_fingerprint": embedding_space,
        "drafts": [draft.model_dump(mode="json") for draft in drafts],
        "vectors": vectors,
    })
    decode_batch(
        value,
        owner_id=owner_id,
        session_id=session_id,
        request_fingerprint=request_fingerprint,
        policy=policy,
        embedding_space=embedding_space,
    )
    return value


def decode_batch(
    value: str, *, owner_id, session_id, request_fingerprint, policy, embedding_space
):
    batch = json.loads(value)
    for key, expected in {
        "protocol": PROTOCOL, "owner_id": owner_id, "session_id": session_id,
        "request_fingerprint": request_fingerprint, "policy_fingerprint": policy,
        "embedding_space_fingerprint": embedding_space,
    }.items():
        if batch.get(key) != expected:
            raise ValueError(f"Prepared batch {key} mismatch; preserve the database")
    drafts = [ClaimDraft.model_validate(item) for item in batch["drafts"]]
    vectors = batch["vectors"]
    if not isinstance(vectors, list) or len(vectors) != len(drafts):
        raise ValueError("Prepared vector count differs from the claim count")
    dimensions = set()
    for draft, vector in zip(drafts, vectors, strict=True):
        if draft.owner_id != owner_id or draft.source_session_id != session_id:
            raise ValueError("Prepared claim belongs to a different owner or session")
        if not isinstance(vector, list) or not vector or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in vector
        ):
            raise ValueError("Prepared vector must contain finite numbers")
        dimensions.add(len(vector))
    if len(dimensions) > 1:
        raise ValueError("Prepared vectors have inconsistent dimensions")
    return drafts, vectors
