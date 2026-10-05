from __future__ import annotations

import pytest

from stacmem.admission import FactualityAdmissionGate
from stacmem.models import AdmissionDecision, ClaimDraft, UpdateKind


def draft(content: str, *, predicate: str = "employment.current") -> ClaimDraft:
    return ClaimDraft(
        owner_id="u1",
        subject="Alice",
        predicate=predicate,
        object_value="Contoso Health",
        assertion_time=2000,
        observed_at=2000,
        valid_start=1500,
        update_kind=UpdateKind.ASSERTION,
        functional=True,
        source_content=content,
    )


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("If I joined Contoso Health, I would leave my current job.", "hypothetical"),
        ("I did not join Contoso Health.", "negated_assertion"),
        ("I might join Contoso Health next year.", "uncertain_or_planned"),
        ('The quote "I joined Contoso Health" is an example.', "quoted_or_reported"),
        (
            "\u5982\u679c\u6211\u52a0\u5165 Contoso Health\uff0c"
            "\u90a3\u53ea\u662f\u5047\u8bbe\u3002",
            "hypothetical",
        ),
    ],
)
def test_nonfactual_signals_are_reason_coded(content: str, reason: str) -> None:
    assessment = FactualityAdmissionGate().assess(draft(content))
    assert assessment.decision == AdmissionDecision.QUARANTINE
    assert reason in assessment.reasons


def test_definite_future_transition_is_accepted() -> None:
    assessment = FactualityAdmissionGate().assess(
        draft("I will join Contoso Health on October 15, 2025.")
    )
    assert assessment.decision == AdmissionDecision.ACCEPT


def test_plan_language_is_valid_for_plan_predicate() -> None:
    assessment = FactualityAdmissionGate().assess(
        draft("I plan to visit Tokyo next month.", predicate="plan.travel")
    )
    assert assessment.decision == AdmissionDecision.ACCEPT


def test_negation_is_valid_for_explicit_retraction() -> None:
    candidate = draft("I did not keep the trip plan.", predicate="plan.travel")
    candidate.update_kind = UpdateKind.RETRACTION
    assessment = FactualityAdmissionGate().assess(candidate)
    assert assessment.decision == AdmissionDecision.ACCEPT
