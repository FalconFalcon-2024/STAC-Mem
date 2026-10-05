from __future__ import annotations

from stacmem.config import AppConfig
from stacmem.grounding import PropositionGrounder
from stacmem.models import Claim, ClaimDraft, ClaimStatus, UpdateKind
from stacmem.pipeline import StacMemory


def draft(
    content: str,
    *,
    value: str = "Contoso Health",
    subject: str = "test_user",
    predicate: str = "employment.current",
    update_kind: UpdateKind = UpdateKind.ASSERTION,
    metadata: dict | None = None,
) -> ClaimDraft:
    return ClaimDraft(
        owner_id="test_user",
        subject=subject,
        predicate=predicate,
        object_value=value,
        assertion_time=2000,
        observed_at=2000,
        valid_start=1500,
        update_kind=update_kind,
        functional=True,
        source_content=content,
        metadata=metadata or {},
    )


def prior(value: str = "Northwind Labs") -> Claim:
    candidate = draft("I work at Northwind Labs.", value=value)
    return Claim.from_draft(candidate, transaction_start=1000, version=1)


def test_scope_is_local_to_the_value_supporting_proposition() -> None:
    certificate = PropositionGrounder().ground(
        draft(
            "I might sound indecisive, yet I accepted a position at Contoso Health."
        ),
        [prior()],
    )
    assert certificate.support_span == "I accepted a position at Contoso Health"
    assert certificate.factuality == "asserted"
    assert certificate.subject_alignment == "aligned"
    assert certificate.transition_entailment is True


def test_negation_of_old_value_does_not_negate_new_value() -> None:
    certificate = PropositionGrounder().ground(
        draft(
            "I do not work at Northwind Labs now; "
            "I accepted a position at Contoso Health."
        ),
        [prior()],
    )
    assert certificate.factuality == "asserted"
    assert certificate.transition_entailment is True


def test_nonfactual_frame_is_attached_to_support_span() -> None:
    certificate = PropositionGrounder().ground(
        draft("In a screenplay, I am employed by Contoso Health."), [prior()]
    )
    assert certificate.factuality == "nonfactual"
    assert "nonfactual_frame" in certificate.factuality_reasons
    assert certificate.transition_entailment is False


def test_third_party_support_is_misaligned() -> None:
    certificate = PropositionGrounder().ground(
        draft(
            "My manager accepted a position at Contoso Health; "
            "I remain at Northwind Labs."
        ),
        [prior()],
    )
    assert certificate.subject_alignment == "misaligned"
    assert certificate.transition_entailment is False


def test_other_object_support_is_misaligned() -> None:
    certificate = PropositionGrounder().ground(
        draft(
            "The office printer was transferred to cabinet B; "
            "the backup drive remains in drawer A.",
            value="cabinet B",
            subject="backup-drive-001",
            predicate="item.location.current",
        ),
        [
            prior("drawer A").model_copy(
                update={"subject": "backup-drive-001", "predicate": "item.location.current"}
            )
        ],
    )
    assert certificate.subject_alignment == "misaligned"


def test_plain_later_assertion_is_not_automatically_a_transition() -> None:
    certificate = PropositionGrounder().ground(
        draft("My employer is Contoso Health."), [prior()]
    )
    assert certificate.factuality == "asserted"
    assert certificate.transition_entailment is False


def test_invalid_structured_hint_is_ignored() -> None:
    certificate = PropositionGrounder().ground(
        draft(
            "My employer is Contoso Health.",
            metadata={
                "proposition_grounding": {
                    "evidence_span": "This span was invented.",
                    "factuality": "nonfactual",
                    "subject_alignment": "misaligned",
                }
            },
        ),
        [prior()],
    )
    assert certificate.structured_hint_used is False
    assert certificate.factuality == "asserted"


def v7_memory(tmp_path) -> StacMemory:
    config = AppConfig()
    config.runtime.database_path = str(tmp_path / "v7.sqlite3")
    config.extraction.provider = "rule"
    config.embedding.provider = "hash"
    config.rerank.provider = "none"
    config.admission.mode = "proposition_v2"
    config.conflict.transition_mode = "proposition_v2"
    return StacMemory.from_app_config(config, _install_contracts=False)


def test_v7_materializes_grounded_transition(tmp_path) -> None:
    memory = v7_memory(tmp_path)
    try:
        old = draft("I work at Northwind Labs.", value="Northwind Labs")
        old.assertion_time = old.observed_at = old.valid_start = 1000
        memory.ingest_drafts([old])
        outcome = memory.ingest_drafts(
            [
                draft(
                    "After ending my tenure at Northwind Labs, "
                    "I accepted a position at Contoso Health."
                )
            ]
        )[0]
        assert outcome.claim.update_kind == UpdateKind.TRANSITION
        assert outcome.claim.status == ClaimStatus.ACTIVE
        assert outcome.claim.metadata["update_kind_normalizer"] == (
            "proposition-transition-v2"
        )
        assert outcome.claim.metadata["proposition_certificate"][
            "transition_entailment"
        ] is True
        assert outcome.relations[0].relation.value == "supersedes"
    finally:
        memory.close()


def test_v7_quarantines_nonfactual_and_misaligned_claims(tmp_path) -> None:
    memory = v7_memory(tmp_path)
    try:
        old = draft("I work at Northwind Labs.", value="Northwind Labs")
        old.assertion_time = old.observed_at = old.valid_start = 1000
        memory.ingest_drafts([old])
        nonfactual = memory.ingest_drafts(
            [draft("Within a simulation, I am employed by Contoso Health.")]
        )[0]
        misaligned = memory.ingest_drafts(
            [
                draft(
                    "My supervisor accepted a position at Fabrikam Studio; "
                    "I remain at Northwind Labs.",
                    value="Fabrikam Studio",
                )
            ]
        )[0]
        assert nonfactual.claim.status == ClaimStatus.QUARANTINED
        assert "nonfactual_frame" in nonfactual.admission_reasons
        assert misaligned.claim.status == ClaimStatus.QUARANTINED
        assert "subject_mismatch" in misaligned.admission_reasons
        assert not nonfactual.relations
        assert not misaligned.relations
    finally:
        memory.close()
