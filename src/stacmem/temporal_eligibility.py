"""Evidence eligibility is not a claim about when an undated state began."""

from __future__ import annotations

from .models import ClaimDraft
from .time_utils import contains, overlaps


def evidence_floor(claim: ClaimDraft) -> int | None:
    marker = claim.metadata.get("temporal_eligibility", {})
    if isinstance(marker, dict) and marker.get("kind") == "unknown_onset":
        # Use authenticated assertion time, never a model-supplied metadata timestamp.
        return claim.assertion_time
    if claim.valid_start is None and claim.valid_end is None:
        return claim.assertion_time
    return None


def evidence_interval(claim: ClaimDraft) -> tuple[int | None, int | None]:
    floor = evidence_floor(claim)
    start = claim.valid_start
    if floor is not None:
        start = floor if start is None else max(start, floor)
    return start, claim.valid_end


def eligible_at(claim: ClaimDraft, point: int) -> bool:
    return contains(*evidence_interval(claim), point)


def evidence_overlaps(left: ClaimDraft, right: ClaimDraft) -> bool:
    return overlaps(*evidence_interval(left), *evidence_interval(right))
