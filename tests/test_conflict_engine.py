from __future__ import annotations

import pytest

from stacmem.models import (
    AdmissionDecision,
    ClaimDraft,
    ClaimStatus,
    Place,
    RelationType,
    UpdateKind,
)
from stacmem.time_utils import ensure_ms


def draft(
    *,
    value: str,
    observed: int,
    start: int,
    kind: UpdateKind = UpdateKind.ASSERTION,
    predicate: str = "residence.current",
    place: Place | None = None,
) -> ClaimDraft:
    return ClaimDraft(
        owner_id="u1",
        subject="Alice",
        predicate=predicate,
        object_value=value,
        assertion_time=observed,
        observed_at=observed,
        valid_start=start,
        place=place,
        update_kind=kind,
        functional=True,
        source_content=value,
    )


def test_transition_closes_old_valid_interval(memory) -> None:
    old = memory.ingest_drafts([draft(value="City A", observed=1000, start=1000)])[0]
    new = memory.ingest_drafts(
        [draft(value="City B", observed=2000, start=2000, kind=UpdateKind.TRANSITION)]
    )[0]
    stored_old = memory.store.get_claims([old.claim.id])[0]
    assert stored_old.status == ClaimStatus.SUPERSEDED
    assert stored_old.valid_end == 2000
    assert new.relations[0].relation == RelationType.SUPERSEDES


def test_same_boundary_transition_never_materializes_zero_length_interval(memory) -> None:
    memory.conflict.temporal_grounding_mode = "certificate_v2"
    boundary = ensure_ms("2025-01-01")
    old_draft = draft(value="City A", observed=1000, start=boundary,
                      kind=UpdateKind.TRANSITION)
    old_draft.source_content = "Since 2025-01-01 I moved to City A."
    new_draft = draft(value="City B", observed=1100, start=boundary,
                      kind=UpdateKind.TRANSITION)
    new_draft.source_content = "Since 2025-01-01 I moved to City B."
    old = memory.ingest_drafts([old_draft])[0]
    new = memory.ingest_drafts([new_draft])[0]

    stored_old = memory.store.get_claims([old.claim.id])[0]

    assert new.relations[0].relation == RelationType.SUPERSEDES
    assert stored_old.status == ClaimStatus.SUPERSEDED
    assert stored_old.valid_start == boundary
    assert stored_old.valid_end is None


def test_observed_at_survives_sqlite_round_trip(memory) -> None:
    outcome = memory.ingest_drafts([draft(value="City A", observed=1234, start=1000)])[0]
    stored = memory.store.get_claims([outcome.claim.id])[0]
    assert stored.observed_at == 1234
    assert stored.transaction_start == 1234


def test_correction_closes_transaction_time(memory) -> None:
    old = memory.ingest_drafts([draft(value="Wrong Co", observed=1000, start=500)])[0]
    new = memory.ingest_drafts(
        [draft(value="Right Co", observed=2000, start=500, kind=UpdateKind.CORRECTION)]
    )[0]
    stored_old = memory.store.get_claims([old.claim.id])[0]
    assert stored_old.status == ClaimStatus.CORRECTED
    assert stored_old.transaction_end == 2000
    assert new.relations[0].relation == RelationType.CORRECTS


def test_spatial_preferences_coexist(memory) -> None:
    tokyo = Place(name="Tokyo", hierarchy=["JP", "Tokyo"], role="scope")
    osaka = Place(name="Osaka", hierarchy=["JP", "Osaka"], role="scope")
    memory.ingest_drafts(
        [
            draft(
                value="Cafe A",
                observed=1000,
                start=1000,
                predicate="preference.coffee_shop",
                place=tokyo,
            )
        ]
    )
    outcome = memory.ingest_drafts(
        [
            draft(
                value="Cafe B",
                observed=2000,
                start=1000,
                predicate="preference.coffee_shop",
                place=osaka,
            )
        ]
    )[0]
    assert outcome.relations[0].relation == RelationType.COEXISTS
    assert all(item.status == ClaimStatus.ACTIVE for item in memory.store.owner_claims("u1"))


def test_retraction_removes_positive_current_value(memory) -> None:
    memory.ingest_drafts(
        [draft(value="Trip A", observed=1000, start=3000, predicate="plan.travel")]
    )
    outcome = memory.ingest_drafts(
        [
            draft(
                value="Trip A",
                observed=2000,
                start=3000,
                kind=UpdateKind.RETRACTION,
                predicate="plan.travel",
            )
        ]
    )[0]
    assert outcome.claim.status == ClaimStatus.RETRACTED
    assert outcome.relations[0].relation == RelationType.RETRACTS


def test_predicate_aliases_share_one_conflict_slot(memory) -> None:
    memory.ingest_drafts(
        [draft(value="Wrong Co", observed=1000, start=500, predicate="job.employer")]
    )
    outcome = memory.ingest_drafts(
        [
            draft(
                value="Right Co",
                observed=2000,
                start=500,
                kind=UpdateKind.CORRECTION,
                predicate="employment.organization",
            )
        ]
    )[0]
    assert outcome.relations[0].relation == RelationType.CORRECTS
    assert outcome.claim.version == 2


def test_batch_ingest_returns_refreshed_statuses(memory) -> None:
    outcomes = memory.ingest_drafts(
        [
            draft(value="Hangzhou", observed=1000, start=1000),
            draft(
                value="Chengdu",
                observed=2000,
                start=2000,
                kind=UpdateKind.TRANSITION,
            ),
        ]
    )
    assert outcomes[0].claim.status == ClaimStatus.SUPERSEDED
    assert outcomes[0].claim.valid_end == 2000
    assert outcomes[1].claim.status == ClaimStatus.ACTIVE


def test_transition_cues_upgrade_llm_assertion_and_close_old_state(memory) -> None:
    old = memory.ingest_drafts(
        [
            draft(
                value="Northwind Labs",
                observed=1000,
                start=500,
                predicate="employment.current",
            )
        ]
    )[0]
    transition = draft(
        value="Contoso Health",
        observed=2000,
        start=1500,
        predicate="employment.current",
    )
    transition.source_content = (
        "I left Northwind Labs and joined Contoso Health as a senior product designer."
    )
    outcome = memory.ingest_drafts([transition])[0]

    stored_old = memory.store.get_claims([old.claim.id])[0]
    assert outcome.claim.predicate == "employment.organization"
    assert outcome.claim.update_kind == UpdateKind.TRANSITION
    assert outcome.claim.metadata["original_update_kind"] == "assertion"
    assert outcome.claim.version == 2
    assert outcome.relations[0].relation == RelationType.SUPERSEDES
    assert stored_old.status == ClaimStatus.SUPERSEDED
    assert stored_old.valid_end == 1500


@pytest.mark.parametrize(
    "source_content",
    [
        "If I joined Contoso Health, I would leave Northwind Labs.",
        "I did not leave Northwind Labs or join Contoso Health.",
        'The quote "I left Northwind Labs and joined Contoso Health" is an example.',
        "I might leave Northwind Labs and join Contoso Health next year.",
        (
            "\u5982\u679c\u6211\u79bb\u5f00 Northwind Labs \u5e76\u52a0\u5165 "
            "Contoso Health\uff0c\u90a3\u53ea\u662f\u4e00\u79cd\u5047\u8bbe\u3002"
        ),
        (
            "\u6211\u6ca1\u6709\u79bb\u5f00 Northwind Labs\uff0c"
            "\u4e5f\u6ca1\u6709\u52a0\u5165 Contoso Health\u3002"
        ),
    ],
)
def test_nonfactual_claims_are_quarantined_without_superseding(
    memory, source_content: str
) -> None:
    old = memory.ingest_drafts(
        [
            draft(
                value="Northwind Labs",
                observed=1000,
                start=500,
                predicate="employment.current",
            )
        ]
    )[0]
    candidate = draft(
        value="Contoso Health",
        observed=2000,
        start=1500,
        predicate="employment.current",
    )
    candidate.source_content = source_content
    outcome = memory.ingest_drafts([candidate])[0]

    stored_old = memory.store.get_claims([old.claim.id])[0]
    assert outcome.claim.update_kind == UpdateKind.ASSERTION
    assert outcome.admission_decision == AdmissionDecision.QUARANTINE
    assert outcome.admission_reasons
    assert outcome.relations == []
    assert outcome.claim.status == ClaimStatus.QUARANTINED
    assert stored_old.status == ClaimStatus.ACTIVE


def test_admission_can_be_disabled_without_enabling_false_supersession(memory) -> None:
    memory.conflict.admission_gate.enabled = False
    old = memory.ingest_drafts(
        [draft(value="Northwind Labs", observed=1000, start=500, predicate="works_at")]
    )[0]
    candidate = draft(
        value="Contoso Health",
        observed=2000,
        start=1500,
        predicate="works_at",
    )
    candidate.source_content = "If I joined Contoso Health, I would leave Northwind Labs."
    outcome = memory.ingest_drafts([candidate])[0]

    stored_old = memory.store.get_claims([old.claim.id])[0]
    assert outcome.admission_decision == AdmissionDecision.ACCEPT
    assert outcome.relations[0].relation == RelationType.CONTRADICTS
    assert outcome.claim.status == ClaimStatus.DISPUTED
    assert stored_old.status == ClaimStatus.DISPUTED


def test_transition_evidence_requires_new_value_in_source(memory) -> None:
    memory.ingest_drafts(
        [draft(value="Northwind Labs", observed=1000, start=500, predicate="works_at")]
    )
    candidate = draft(
        value="Contoso Health",
        observed=2000,
        start=1500,
        predicate="works_at",
    )
    candidate.source_content = "My colleague left Northwind Labs and joined another company."
    outcome = memory.ingest_drafts([candidate])[0]

    assert outcome.claim.update_kind == UpdateKind.ASSERTION
    assert outcome.relations[0].relation == RelationType.CONTRADICTS


def test_transition_evidence_inference_can_be_disabled(memory) -> None:
    memory.conflict.infer_transition_from_evidence = False
    memory.ingest_drafts(
        [draft(value="Northwind Labs", observed=1000, start=500, predicate="works_at")]
    )
    candidate = draft(
        value="Contoso Health",
        observed=2000,
        start=1500,
        predicate="works_at",
    )
    candidate.source_content = "I left Northwind Labs and joined Contoso Health."
    outcome = memory.ingest_drafts([candidate])[0]

    assert outcome.claim.update_kind == UpdateKind.ASSERTION
    assert outcome.relations[0].relation == RelationType.CONTRADICTS


def test_chinese_transition_cue_upgrades_assertion(memory) -> None:
    memory.ingest_drafts(
        [draft(value="Hangzhou", observed=1000, start=500, predicate="residence.current")]
    )
    candidate = draft(
        value="Chengdu",
        observed=2000,
        start=1500,
        predicate="residence.current",
    )
    candidate.source_content = (
        "\u6211\u5df2\u7ecf\u4ece Hangzhou \u642c\u5230 Chengdu\uff0c"
        "\u73b0\u5728\u4f4f\u5728 Chengdu\u3002"
    )
    outcome = memory.ingest_drafts([candidate])[0]

    assert outcome.claim.update_kind == UpdateKind.TRANSITION
    assert outcome.relations[0].relation == RelationType.SUPERSEDES


def test_quarantined_claim_is_auditable_but_does_not_pollute_future_transition(memory) -> None:
    old = memory.ingest_drafts(
        [draft(value="Northwind Labs", observed=1000, start=500, predicate="works_at")]
    )[0]
    unsafe = draft(
        value="Fabrikam Studio",
        observed=1500,
        start=1200,
        predicate="works_at",
    )
    unsafe.source_content = "I might leave Northwind Labs and join Fabrikam Studio."
    quarantined = memory.ingest_drafts([unsafe])[0]
    factual = draft(
        value="Contoso Health",
        observed=2000,
        start=1500,
        predicate="works_at",
    )
    factual.source_content = "I left Northwind Labs and joined Contoso Health."
    accepted = memory.ingest_drafts([factual])[0]

    stored = {
        item.id: item
        for item in memory.store.get_claims(
            [old.claim.id, quarantined.claim.id, accepted.claim.id]
        )
    }
    assert quarantined.claim.version == 2
    assert quarantined.claim.status == ClaimStatus.QUARANTINED
    assert accepted.claim.version == 3
    assert len(accepted.relations) == 1
    assert accepted.relations[0].target_claim_id == old.claim.id
    assert accepted.relations[0].relation == RelationType.SUPERSEDES
    assert stored[old.claim.id].status == ClaimStatus.SUPERSEDED
    assert stored[quarantined.claim.id].status == ClaimStatus.QUARANTINED


def test_job_title_aliases_share_transition_slot(memory) -> None:
    old = memory.ingest_drafts(
        [
            draft(
                value="product designer",
                observed=1000,
                start=500,
                predicate="occupation.current",
            )
        ]
    )[0]
    transition = draft(
        value="senior product designer",
        observed=2000,
        start=1500,
        predicate="employment.position",
    )
    transition.source_content = "I joined Contoso Health as a senior product designer."
    outcome = memory.ingest_drafts([transition])[0]

    stored_old = memory.store.get_claims([old.claim.id])[0]
    assert outcome.claim.predicate == "employment.position"
    assert outcome.claim.version == 2
    assert outcome.relations[0].relation == RelationType.SUPERSEDES
    assert stored_old.status == ClaimStatus.SUPERSEDED
