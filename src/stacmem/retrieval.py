"""Hybrid claim recall and query-adaptive spatio-temporal scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import RetrievalConfig
from .embeddings import Embedder, cosine_similarity
from .models import (
    Claim,
    ClaimStatus,
    QueryFrame,
    ScoreBreakdown,
    ScoredClaim,
    TemporalIntent,
    normalize_text,
)
from .rerank import Reranker
from .spatial import spatial_similarity
from .store import ClaimStore
from .temporal_eligibility import eligible_at, evidence_interval
from .time_utils import contains, interval_distance_ms, now_ms, overlaps, temporal_decay

_TOKEN = re.compile(r"[\w\-]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    temporal: bool = False
    spatial: bool = False
    conflict: bool = False


VARIANTS: dict[str, RetrievalVariant] = {
    "semantic": RetrievalVariant("semantic"),
    "temporal": RetrievalVariant("temporal", temporal=True),
    "spatial": RetrievalVariant("spatial", spatial=True),
    "spatiotemporal": RetrievalVariant("spatiotemporal", temporal=True, spatial=True),
    "conflict": RetrievalVariant("conflict", conflict=True),
    "full": RetrievalVariant("full", temporal=True, spatial=True, conflict=True),
}


class ClaimRetriever:
    def __init__(
        self,
        store: ClaimStore,
        embedder: Embedder,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.reranker = reranker

    def retrieve(
        self,
        frame: QueryFrame,
        *,
        variant: RetrievalVariant,
        top_k: int | None = None,
    ) -> list[ScoredClaim]:
        limit = max(top_k or self.config.top_k, self.config.candidate_pool)
        claims = [
            claim
            for claim in self.store.owner_claims(frame.owner_id)
            if claim.status != ClaimStatus.QUARANTINED
        ]
        if not claims:
            return []

        has_structured_target = (
            frame.target_subject is not None or frame.target_predicate is not None
        )
        if has_structured_target:
            structural = [claim for claim in claims if _matches_query_slot(claim, frame)]
            if structural:
                claims = structural
                frame.metadata = {
                    **frame.metadata,
                    "retrieval_route": "structured_slot",
                    "routed_candidate_count": len(claims),
                }
            elif frame.metadata.get("compiler"):
                frame.metadata = {
                    **frame.metadata,
                    "retrieval_route": "semantic_fallback_no_slot_match",
                    "routed_candidate_count": len(claims),
                }
            else:
                claims = []
                frame.metadata = {
                    **frame.metadata,
                    "retrieval_route": "structured_slot_empty",
                    "routed_candidate_count": 0,
                }
        else:
            frame.metadata = {
                **frame.metadata,
                "retrieval_route": "semantic_global",
                "routed_candidate_count": len(claims),
            }
        if not claims:
            return []

        query_vector = self.embedder.embed([frame.raw_query])[0]
        semantic = {
            claim.id: max(0.0, (cosine_similarity(query_vector, claim.vector) + 1.0) / 2.0)
            for claim in claims
        }
        lexical_rows = self.store.lexical_search(frame.owner_id, frame.raw_query, limit=limit)
        lexical = _normalize_lexical(lexical_rows)

        pre_ranked = sorted(
            claims,
            key=lambda item: max(semantic.get(item.id, 0.0), lexical.get(item.id, 0.0)),
            reverse=True,
        )[:limit]
        if self.reranker is not None and pre_ranked:
            rerank_values = self.reranker.score(
                frame.raw_query,
                [f"{claim.embedding_text()} {claim.source_content}" for claim in pre_ranked],
            )
            semantic.update(
                {
                    claim.id: 0.5 * semantic.get(claim.id, 0.0) + 0.5 * rerank_score
                    for claim, rerank_score in zip(pre_ranked, rerank_values, strict=True)
                }
            )
        weights = self._effective_weights(frame, variant)
        scored = [
            self._score(
                claim,
                frame,
                semantic.get(claim.id, 0.0),
                lexical.get(claim.id, 0.0),
                weights,
                variant,
            )
            for claim in pre_ranked
        ]
        scored.sort(key=lambda item: (item.score.total, item.claim.version), reverse=True)
        return scored[:limit]

    def _score(
        self,
        claim: Claim,
        frame: QueryFrame,
        semantic: float,
        lexical: float,
        weights: dict[str, float],
        variant: RetrievalVariant,
    ) -> ScoredClaim:
        temporal = _temporal_score(claim, frame) if variant.temporal else 0.0
        spatial = spatial_similarity(claim.place, frame.place) if variant.spatial else 0.0
        validity = _validity_score(claim, frame) if variant.conflict else 0.0
        penalty = 0.15 if variant.conflict and claim.status == ClaimStatus.DISPUTED else 0.0
        total = (
            weights["semantic"] * semantic
            + weights["lexical"] * lexical
            + weights["temporal"] * temporal
            + weights["spatial"] * spatial
            + weights["validity"] * validity
            + weights["confidence"] * claim.confidence
            - penalty
        )
        return ScoredClaim(
            claim=claim,
            score=ScoreBreakdown(
                semantic=round(semantic, 6),
                lexical=round(lexical, 6),
                temporal=round(temporal, 6),
                spatial=round(spatial, 6),
                validity=round(validity, 6),
                confidence=round(claim.confidence, 6),
                conflict_penalty=penalty,
                total=round(total, 6),
            ),
        )

    def _effective_weights(self, frame: QueryFrame, variant: RetrievalVariant) -> dict[str, float]:
        base = self.config.weights.model_dump()
        enabled = {
            "semantic": True,
            "lexical": True,
            "temporal": variant.temporal and frame.temporal_intent != TemporalIntent.UNSPECIFIED,
            "spatial": variant.spatial and frame.place is not None,
            "validity": variant.conflict,
            "confidence": True,
        }
        if frame.temporal_intent in {TemporalIntent.CURRENT, TemporalIntent.AS_OF}:
            base["temporal"] *= 1.35
            base["validity"] *= 1.25
        if frame.temporal_intent == TemporalIntent.HISTORY:
            base["temporal"] *= 1.15
            base["validity"] *= 0.4
        if frame.place is not None:
            base["spatial"] *= 1.35
        active = {key: value for key, value in base.items() if enabled.get(key, False)}
        total = sum(active.values()) or 1.0
        return {key: (active.get(key, 0.0) / total) for key in base}


def _normalize_lexical(rows: list[tuple[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    raw = [score for _, score in rows]
    high = max(raw)
    low = min(raw)
    if high <= low:
        return {claim_id: 1.0 for claim_id, _ in rows}
    return {claim_id: (score - low) / (high - low) for claim_id, score in rows}


def _temporal_score(claim: Claim, frame: QueryFrame) -> float:
    start, end = evidence_interval(claim)
    if frame.temporal_intent == TemporalIntent.HISTORY:
        return 1.0
    if frame.temporal_intent == TemporalIntent.INTERVAL:
        if overlaps(
            start,
            end,
            frame.interval_start,
            frame.interval_end,
        ):
            return 1.0
        anchor = frame.interval_start or frame.interval_end or now_ms()
        return temporal_decay(interval_distance_ms(start, end, anchor))
    point = frame.query_time
    if point is None:
        return 0.5
    distance = interval_distance_ms(start, end, point)
    return temporal_decay(distance)


def _validity_score(claim: Claim, frame: QueryFrame) -> float:
    known_at = frame.knowledge_time or now_ms()
    if not contains(claim.transaction_start, claim.transaction_end, known_at):
        return 0.0
    if frame.query_time is not None and not eligible_at(claim, frame.query_time):
        return 0.0
    if frame.temporal_intent == TemporalIntent.INTERVAL and not overlaps(
        *evidence_interval(claim), frame.interval_start, frame.interval_end
    ):
        return 0.0
    return {
        ClaimStatus.ACTIVE: 1.0,
        ClaimStatus.SUPERSEDED: 0.75,
        ClaimStatus.DISPUTED: 0.25,
        ClaimStatus.CORRECTED: 0.0,
        ClaimStatus.RETRACTED: 0.0,
    }[claim.status]


def lexical_overlap(query: str, claim: Claim) -> float:
    query_tokens = set(_TOKEN.findall(normalize_text(query)))
    claim_tokens = set(_TOKEN.findall(normalize_text(claim.embedding_text())))
    if not query_tokens or not claim_tokens:
        return 0.0
    return len(query_tokens & claim_tokens) / len(query_tokens | claim_tokens)


def _matches_query_slot(claim: Claim, frame: QueryFrame) -> bool:
    subject_matches = frame.target_subject is None or normalize_text(
        frame.target_subject
    ) == normalize_text(claim.subject)
    predicate_matches = frame.target_predicate is None or normalize_text(
        frame.target_predicate
    ) == normalize_text(claim.predicate)
    return subject_matches and predicate_matches
