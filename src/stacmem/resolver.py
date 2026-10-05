"""Deterministic query-time state projection and context shaping."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .context_budget import estimate_tokens
from .models import (
    ClaimStatus,
    EvidencePack,
    QueryFrame,
    ScoredClaim,
    TemporalIntent,
)
from .retrieval import RetrievalVariant
from .store import ClaimStore
from .temporal_eligibility import eligible_at, evidence_floor, evidence_interval
from .time_utils import contains, now_ms, overlaps, to_iso


class StateResolver:
    def __init__(self, store: ClaimStore, *, token_budget: int = 4000) -> None:
        self.store = store
        self.token_budget = token_budget

    def resolve(
        self,
        frame: QueryFrame,
        candidates: list[ScoredClaim],
        *,
        variant: RetrievalVariant,
        top_k: int | None = None,
    ) -> EvidencePack:
        output_limit = max(top_k or 10, 1)
        candidates = [
            item for item in candidates if item.claim.status != ClaimStatus.QUARANTINED
        ]
        if not variant.conflict:
            candidates = candidates[:output_limit]
            for item in candidates:
                item.decision = "ranked_without_state_resolution"
            pack = EvidencePack(
                query=frame,
                claims=candidates,
                diagnostics={
                    "resolver": "disabled",
                    "input_candidates": len(candidates),
                    "query_routing": frame.metadata.get("retrieval_route", "unknown"),
                },
            )
            pack.context = compose_context(pack, self.token_budget)
            return pack

        known_at = frame.knowledge_time or now_ms()
        by_slot: dict[str, list[ScoredClaim]] = defaultdict(list)
        for item in candidates:
            by_slot[item.claim.slot_key].append(item)

        selected: list[ScoredClaim] = []
        warnings: list[str] = []
        suppressed = 0
        for slot_items in by_slot.values():
            slot_items.sort(key=lambda item: (item.claim.version, item.score.total), reverse=True)
            eligible = [
                item
                for item in slot_items
                if contains(
                    item.claim.transaction_start,
                    item.claim.transaction_end,
                    known_at,
                )
            ]
            if frame.temporal_intent == TemporalIntent.HISTORY:
                for item in eligible:
                    item.decision = "history_version"
                selected.extend(eligible)
                continue

            if frame.query_time is not None:
                eligible = [
                    item
                    for item in eligible
                    if eligible_at(item.claim, frame.query_time)
                ]
            if frame.temporal_intent == TemporalIntent.INTERVAL:
                eligible = [
                    item for item in eligible if overlaps(
                        *evidence_interval(item.claim), frame.interval_start, frame.interval_end
                    )
                ]

            eligible = [
                item
                for item in eligible
                if not (
                    item.claim.status in {ClaimStatus.CORRECTED, ClaimStatus.RETRACTED}
                    and item.claim.transaction_end is None
                )
            ]
            if not eligible:
                continue

            active = [item for item in eligible if item.claim.status != ClaimStatus.SUPERSEDED]
            if not active:
                active = eligible
            disputed = [item for item in active if item.claim.status == ClaimStatus.DISPUTED]
            distinct_values = {item.claim.object_norm for item in disputed}
            if len(distinct_values) > 1:
                warnings.append(
                    f"unresolved conflict for {active[0].claim.subject}/{active[0].claim.predicate}"
                )
                for item in disputed:
                    item.decision = "unresolved_conflict"
                selected.extend(disputed)
                suppressed += max(0, len(slot_items) - len(disputed))
                continue

            # State eligibility is already enforced above. Ranking decides
            # among simultaneously valid coexisting values (notably a
            # place-scoped preference); version is only the tie-breaker.
            winner = max(active, key=lambda item: (item.score.total, item.claim.version))
            winner.decision = "active_state"
            selected.append(winner)
            for item in slot_items:
                if item.claim.id != winner.claim.id:
                    item.decision = "suppressed_version"
                    item.suppressed_by = winner.claim.id
                    suppressed += 1

        selected.sort(key=lambda item: item.score.total, reverse=True)
        if frame.expected_cardinality == "one" and selected:
            best_score = selected[0].score.total
            selected = [
                item
                for item in selected
                if item.decision == "unresolved_conflict" or item.score.total >= best_score - 0.08
            ]
        selected = selected[:output_limit]

        pack = EvidencePack(
            query=frame,
            claims=selected,
            warnings=warnings,
            diagnostics={
                "resolver": "bitemporal-conflict-v1",
                "input_candidates": len(candidates),
                "selected_claims": len(selected),
                "suppressed_claims": suppressed,
                "unresolved_slots": len(warnings),
                "query_routing": frame.metadata.get("retrieval_route", "unknown"),
            },
        )
        pack.context = compose_context(pack, self.token_budget)
        return pack


def compose_context(
    pack: EvidencePack,
    token_budget: int,
    *,
    footer_lines: Iterable[str] = (),
) -> str:
    lines = [
        "# Conflict-aware memory evidence",
    ]
    truncated = False

    def append_if_fits(line: str) -> None:
        nonlocal truncated
        if estimate_tokens("\n".join([*lines, line])) <= token_budget:
            lines.append(line)
        else:
            truncated = True

    append_if_fits(f"Query temporal intent: {pack.query.temporal_intent.value}")
    if pack.query.query_time is not None:
        append_if_fits(f"Query valid time: {to_iso(pack.query.query_time)}")
    if pack.query.knowledge_time is not None:
        append_if_fits(f"Knowledge cutoff: {to_iso(pack.query.knowledge_time)}")
    for warning in pack.warnings:
        append_if_fits(f"WARNING: {warning}; do not silently choose one value.")

    for index, item in enumerate(pack.claims, start=1):
        claim = item.claim
        place = claim.place.name if claim.place and claim.place.name else "unspecified"
        floor = evidence_floor(claim)
        start_label = to_iso(claim.valid_start) or ("unknown" if floor is not None else "-inf")
        eligibility = (
            f"evidence_from={to_iso(floor)} (not an onset); " if floor is not None else ""
        )
        line = (
            f"[C{index}] {claim.subject} | {claim.predicate} | {claim.object_value} "
            f"[valid={start_label}..{to_iso(claim.valid_end) or '+inf'}; "
            f"{eligibility}observed={to_iso(claim.transaction_start)}; place={place}; "
            f"state={item.decision}; source_session={claim.source_session_id or 'unknown'}]"
        )
        if claim.source_content:
            line += f" Evidence: {claim.source_content}"
        if estimate_tokens("\n".join([*lines, line])) > token_budget:
            truncated = True
            break
        lines.append(line)

    for line in footer_lines:
        if estimate_tokens("\n".join([*lines, line])) <= token_budget:
            lines.append(line)
        else:
            truncated = True

    context = "\n".join(lines)
    pack.diagnostics.update(
        {
            "context_budget_method": "multilingual-estimate-v1",
            "context_token_budget": token_budget,
            "context_estimated_tokens": estimate_tokens(context),
            "context_truncated": truncated
            or len(pack.claims) > sum(line.startswith("[C") for line in lines),
        }
    )
    return context
