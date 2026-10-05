"""Auditable pre-ledger factuality admission for extracted claim drafts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from .grounding import PropositionCertificate, PropositionGrounder
from .models import AdmissionDecision, Claim, ClaimDraft, UpdateKind, normalize_text


@dataclass(frozen=True)
class AdmissionAssessment:
    decision: AdmissionDecision
    reasons: tuple[str, ...] = ()
    detector: str = "factuality-admission-v1"
    certificate: PropositionCertificate | None = None


class FactualityAdmissionGate:
    """Conservative lexical gate that quarantines clearly non-asserted content."""

    _english_patterns: ClassVar[tuple[tuple[str, re.Pattern[str]], ...]] = (
        ("hypothetical", re.compile(r"\b(?:if|unless|would)\b")),
        (
            "negated_assertion",
            re.compile(
                r"\b(?:did not|didn't|do not|don't|has not|hasn't|"
                r"have not|haven't|never)\b"
            ),
        ),
        (
            "uncertain_or_planned",
            re.compile(
                r"\b(?:may|might|could|plan(?:s|ned|ning)? to|"
                r"consider(?:s|ed|ing)?|thinking about|hope(?:s|d)? to|"
                r"want(?:s|ed)? to)\b"
            ),
        ),
        (
            "quoted_or_reported",
            re.compile(r"\b(?:hypothetical|example|quotation|quote|rumou?r|allegedly)\b"),
        ),
    )
    _chinese_markers: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("hypothetical", ("如果", "假如", "要是", "假设")),
        ("negated_assertion", ("没有", "并未", "从未")),
        ("uncertain_or_planned", ("计划", "打算", "考虑", "可能", "希望")),
        ("quoted_or_reported", ("例子", "引用", "听说", "据说")),
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "lexical_v1",
        grounder: PropositionGrounder | None = None,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self.grounder = grounder or PropositionGrounder()

    def assess(
        self,
        draft: ClaimDraft,
        *,
        existing: Sequence[Claim] = (),
    ) -> AdmissionAssessment:
        if self.mode == "proposition_v2":
            certificate = self.grounder.ground(draft, existing)
            draft.metadata = {
                **draft.metadata,
                "proposition_certificate": certificate.model_dump(),
            }
            if not self.enabled:
                return AdmissionAssessment(
                    AdmissionDecision.ACCEPT,
                    detector="proposition-admission-v2-disabled",
                    certificate=certificate,
                )
            reasons: list[str] = []
            if not certificate.object_grounded:
                reasons.append("unsupported_object")
            if certificate.subject_alignment == "misaligned":
                reasons.append("subject_mismatch")
            elif certificate.subject_alignment == "unknown":
                reasons.append("subject_unverified")
            if certificate.factuality != "asserted":
                reasons.extend(certificate.factuality_reasons)
            if not certificate.current_state_eligible:
                reasons.extend(certificate.current_state_reasons)
            if reasons:
                return AdmissionAssessment(
                    AdmissionDecision.QUARANTINE,
                    tuple(dict.fromkeys(reasons)),
                    "proposition-admission-v2",
                    certificate,
                )
            return AdmissionAssessment(
                AdmissionDecision.ACCEPT,
                detector="proposition-admission-v2",
                certificate=certificate,
            )

        if self.mode != "lexical_v1":
            raise ValueError(f"unsupported admission mode: {self.mode}")
        if not self.enabled or not draft.source_content.strip():
            return AdmissionAssessment(AdmissionDecision.ACCEPT)
        reasons = detect_nonfactual_signals(draft)
        if reasons:
            return AdmissionAssessment(AdmissionDecision.QUARANTINE, tuple(reasons))
        return AdmissionAssessment(AdmissionDecision.ACCEPT)


def detect_nonfactual_signals(draft: ClaimDraft) -> list[str]:
    """Return stable reason codes; explicit correction/retraction may contain negation."""
    source = normalize_text(draft.source_content)
    reasons: list[str] = []
    for reason, pattern in FactualityAdmissionGate._english_patterns:
        if pattern.search(source):
            reasons.append(reason)
    for reason, markers in FactualityAdmissionGate._chinese_markers:
        if any(marker in source for marker in markers):
            reasons.append(reason)

    if draft.update_kind in {UpdateKind.CORRECTION, UpdateKind.RETRACTION}:
        reasons = [reason for reason in reasons if reason != "negated_assertion"]
    if draft.predicate.casefold().strip().startswith("plan."):
        reasons = [reason for reason in reasons if reason != "uncertain_or_planned"]
    return list(dict.fromkeys(reasons))
