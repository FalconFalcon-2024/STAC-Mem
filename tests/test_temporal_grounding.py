from __future__ import annotations

import pytest

from stacmem.config import AppConfig
from stacmem.models import ClaimDraft, QueryFrame, RelationType, TemporalIntent
from stacmem.pipeline import StacMemory
from stacmem.temporal_grounding import TemporalGrounder
from stacmem.time_utils import ensure_ms


def timestamp(value: str) -> int:
    parsed = ensure_ms(value)
    assert parsed is not None
    return parsed


def draft(
    *,
    value: str,
    source: str,
    observed: int,
    start: int | None,
    end: int | None,
) -> ClaimDraft:
    return ClaimDraft(
        owner_id="u1",
        subject="u1",
        predicate="residence.current",
        object_value=value,
        assertion_time=observed,
        observed_at=observed,
        valid_start=start,
        valid_end=end,
        confidence=1.0,
        functional=True,
        source_session_id=f"session-{observed}",
        source_message_ids=[f"message-{observed}"],
        source_content=source,
        extractor="temporal-test",
    )


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("Before 2025-09-24, I lived in Suzhou.", "Suzhou"),
        ("在 2025-09-24 之前，我住在 Suzhou。", "Suzhou"),
    ],
)
def test_upper_bound_cue_repairs_reversed_model_interval(source: str, value: str) -> None:
    boundary = timestamp("2025-09-24T00:00:00Z")
    item = draft(
        value=value,
        source=source,
        observed=timestamp("2025-10-24T12:00:00Z"),
        start=boundary,
        end=None,
    )

    certificate = TemporalGrounder().ground_and_apply(item)

    assert item.valid_start is None
    assert item.valid_end == boundary
    assert certificate.direction == "upper_bound"
    assert certificate.consistency == "corrected"
    assert certificate.normalization_applied is True
    assert item.metadata["original_temporal_bounds"] == {
        "valid_start": boundary,
        "valid_end": None,
    }


def test_lower_bound_cue_preserves_consistent_interval() -> None:
    boundary = timestamp("2025-09-24T00:00:00Z")
    item = draft(
        value="Tokyo",
        source="从 2025-09-24 起，我现在住在 Tokyo。",
        observed=timestamp("2025-09-24T12:00:00Z"),
        start=boundary,
        end=None,
    )

    certificate = TemporalGrounder().ground_and_apply(item)

    assert item.valid_start == boundary
    assert item.valid_end is None
    assert certificate.direction == "lower_bound"
    assert certificate.consistency == "consistent"
    assert certificate.normalization_applied is False


def test_ambiguous_or_ungrounded_temporal_source_clears_model_bounds() -> None:
    original = timestamp("2025-09-24T00:00:00Z")
    ambiguous = draft(
        value="Suzhou",
        source=(
            "Before 2025-09-24 I lived in Suzhou, but since 2025-10-01 "
            "the record has been disputed."
        ),
        observed=timestamp("2025-10-24T12:00:00Z"),
        start=original,
        end=None,
    )
    ungrounded = draft(
        value="Ningbo",
        source="Before 2025-09-24 I lived in Suzhou.",
        observed=timestamp("2025-10-24T12:00:00Z"),
        start=original,
        end=None,
    )

    ambiguous_certificate = TemporalGrounder().ground_and_apply(ambiguous)
    ungrounded_certificate = TemporalGrounder().ground_and_apply(ungrounded)

    assert ambiguous_certificate.consistency == "ambiguous"
    assert ungrounded_certificate.consistency == "unsupported"
    assert ambiguous.valid_start is None and ambiguous.valid_end is None
    assert ungrounded.valid_start is None and ungrounded.valid_end is None
    assert ambiguous_certificate.detector == "temporal-grounding-v1"
    assert ambiguous_certificate.normalization_applied is True


def test_shared_lower_bound_is_bound_to_arrival_and_departure_propositions() -> None:
    boundary = timestamp("2025-04-17T00:00:00Z")
    source = (
        "\u4ece 2025-04-17 \u8d77\uff0c\u6211\u7ed3\u675f\u4e86\u5728 "
        "Contoso Health \u7684\u5de5\u4f5c\uff0c"
        "\u6b63\u5f0f\u5230 Northwind Labs \u62a5\u5230\u3002"
    )
    arrival = draft(
        value="Northwind Labs",
        source=source,
        observed=timestamp("2025-04-17T12:00:00Z"),
        start=boundary,
        end=None,
    )
    arrival.metadata["proposition_grounding"] = {
        "evidence_span": "\u6b63\u5f0f\u5230 Northwind Labs \u62a5\u5230"
    }
    departure = draft(
        value="Contoso Health",
        source=source,
        observed=timestamp("2025-04-17T12:00:00Z"),
        start=None,
        end=boundary,
    )
    departure.metadata["proposition_grounding"] = {
        "evidence_span": "\u7ed3\u675f\u4e86\u5728 Contoso Health \u7684\u5de5\u4f5c"
    }

    arrival_certificate = TemporalGrounder().ground_and_apply(arrival)
    departure_certificate = TemporalGrounder().ground_and_apply(departure)

    assert arrival_certificate.direction == "lower_bound"
    assert arrival.valid_start == boundary and arrival.valid_end is None
    assert departure_certificate.direction == "upper_bound"
    assert departure.valid_start is None and departure.valid_end == boundary
    assert "departure_proposition_inverts_lower_bound" in departure_certificate.reasons
    assert departure_certificate.temporal_cue_span == "\u4ece 2025-04-17 \u8d77"
    assert departure_certificate.value_support_span == (
        "\u6211\u7ed3\u675f\u4e86\u5728 Contoso Health \u7684\u5de5\u4f5c"
    )


@pytest.mark.parametrize(
    ("source", "old_value", "new_value"),
    [
        (
            "Starting on 2026-07-08, I am scheduled to leave Old Works and join New Labs.",
            "Old Works",
            "New Labs",
        ),
        (
            "2026-07-08 \u8d77\uff0c\u6211\u4ece Old Works \u79bb\u804c"
            "\u5e76\u52a0\u5165 New Labs\u3002",
            "Old Works",
            "New Labs",
        ),
    ],
)
def test_v2_binds_shared_boundary_by_nearest_proposition_role(
    source: str, old_value: str, new_value: str
) -> None:
    boundary = timestamp("2026-07-08T00:00:00Z")
    old = draft(
        value=old_value,
        source=source,
        observed=timestamp("2026-07-01T12:00:00Z"),
        start=None,
        end=boundary,
    )
    old.metadata["proposition_grounding"] = {"evidence_span": source}
    new = draft(
        value=new_value,
        source=source,
        observed=timestamp("2026-07-01T12:00:00Z"),
        start=boundary,
        end=None,
    )
    new.metadata["proposition_grounding"] = {"evidence_span": source}

    old_certificate = TemporalGrounder().ground_and_apply(old, mode="certificate_v2")
    new_certificate = TemporalGrounder().ground_and_apply(new, mode="certificate_v2")

    assert old.valid_start is None and old.valid_end == boundary
    assert old_certificate.direction == "upper_bound"
    assert new.valid_start == boundary and new.valid_end is None
    assert new_certificate.direction == "lower_bound"
    assert "arrival_proposition_preserves_lower_bound" in new_certificate.reasons


@pytest.mark.parametrize(
    "source",
    [
        "Between 2026-04-21 and 2026-06-20, my residence was Kyoto.",
        "\u65e7\u6863\u6848\u8bb0\u8f7d\uff0c2026-04-21 \u81f3 2026-06-20 \u671f\u95f4\uff0c"
        "\u6211\u7684\u4f4f\u5740\u662f Kyoto\u3002",
    ],
)
def test_v2_normalizes_additional_closed_interval_forms(source: str) -> None:
    start = timestamp("2026-04-21T00:00:00Z")
    end = timestamp("2026-06-20T00:00:00Z")
    item = draft(
        value="Kyoto",
        source=source,
        observed=timestamp("2026-07-20T12:00:00Z"),
        start=start,
        end=timestamp("2026-06-21T00:00:00Z"),
    )

    certificate = TemporalGrounder().ground_and_apply(item, mode="certificate_v2")

    assert item.valid_start == start and item.valid_end == end
    assert certificate.direction == "closed_interval"
    assert certificate.consistency == "corrected"


def test_v2_rejects_non_positive_normalization_and_clears_model_bounds() -> None:
    original_start = timestamp("2026-05-01T00:00:00Z")
    original_end = timestamp("2026-06-01T00:00:00Z")
    item = draft(
        value="Old Works",
        source="From 2026-07-08 to 2026-06-08, I worked at Old Works.",
        observed=timestamp("2026-07-01T12:00:00Z"),
        start=original_start,
        end=original_end,
    )

    certificate = TemporalGrounder().ground_and_apply(item, mode="certificate_v2")

    assert item.valid_start is None and item.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.normalization_applied is True
    assert "normalization_rejected_non_positive_interval" in certificate.reasons


def test_temporal_cue_in_another_sentence_does_not_rewrite_claim() -> None:
    boundary = timestamp("2025-04-17T00:00:00Z")
    item = draft(
        value="Contoso Health",
        source=(
            "Since 2025-04-17 I have worked at Northwind Labs. "
            "An old note mentions Contoso Health."
        ),
        observed=timestamp("2025-04-17T12:00:00Z"),
        start=None,
        end=boundary,
    )

    certificate = TemporalGrounder().ground_and_apply(item)

    assert certificate.consistency == "unsupported"
    assert certificate.reasons[0] == "temporal_cue_not_colocated_with_value_evidence"
    assert item.valid_start is None and item.valid_end is None


@pytest.mark.parametrize("field", ["start", "end"])
def test_v2_unsupported_model_bound_cannot_enter_valid_time(field: str) -> None:
    invented = timestamp("2030-01-01T00:00:00Z")
    item = draft(
        value="Aster", source="I work at Aster.",
        observed=timestamp("2024-10-01T12:00:00Z"),
        start=invented if field == "start" else None,
        end=invented if field == "end" else None,
    )

    certificate = TemporalGrounder().ground_and_apply(item, mode="certificate_v2")

    assert item.valid_start is None and item.valid_end is None
    assert certificate.detector == "temporal-grounding-v2"
    assert certificate.consistency == "insufficient"
    assert certificate.normalization_applied is True
    assert item.metadata["original_temporal_bounds"][f"valid_{field}"] == invented


def test_date_in_contrasted_prior_clause_cannot_date_new_value() -> None:
    item = draft(
        value="London",
        source="Since 2024-08-01 I lived in Paris, but now I live in London.",
        observed=timestamp("2025-01-01T12:00:00Z"),
        start=timestamp("2030-01-01T00:00:00Z"), end=None,
    )

    certificate = TemporalGrounder().ground_and_apply(item, mode="certificate_v2")

    assert certificate.consistency == "unsupported"
    assert item.valid_start is None and item.valid_end is None


@pytest.mark.parametrize("source,expected", [
    ("On 2024-08-01 I joined Aster.", "lower_bound"),
    ("On 2024-08-01 I work at Aster.", "observation"),
])
def test_on_date_distinguishes_state_observation_from_onset(source: str, expected: str) -> None:
    item = draft(
        value="Aster", source=source,
        observed=timestamp("2024-08-01T12:00:00Z"),
        start=timestamp("2030-01-01T00:00:00Z"), end=None,
    )

    certificate = TemporalGrounder().ground_and_apply(item, mode="certificate_v2")

    assert certificate.direction == expected
    assert item.valid_start == timestamp("2024-08-01T00:00:00Z")
    assert item.valid_end == (
        timestamp("2024-08-02T00:00:00Z") if expected == "observation" else None
    )


def build_memory(tmp_path, *, grounding: bool, relation_mode: str) -> StacMemory:
    config = AppConfig()
    config.runtime.database_path = str(tmp_path / f"{grounding}-{relation_mode}.sqlite3")
    config.extraction.provider = "rule"
    config.embedding.provider = "hash"
    config.embedding.dimensions = 128
    config.rerank.provider = "none"
    config.temporal_grounding.enabled = grounding
    config.conflict.temporal_relation_mode = relation_mode
    return StacMemory.from_app_config(config, _install_contracts=False)


@pytest.mark.parametrize(
    ("relation_mode", "expected"),
    [
        ("transaction_fallback_v1", RelationType.SUPERSEDES),
        ("valid_interval_v2", RelationType.COEXISTS),
    ],
)
def test_bounded_late_history_does_not_use_transaction_recency(
    tmp_path, relation_mode: str, expected: RelationType
) -> None:
    boundary = timestamp("2025-09-29T00:00:00Z")
    memory = build_memory(tmp_path, grounding=False, relation_mode=relation_mode)
    try:
        memory.ingest_drafts(
            [
                draft(
                    value="Chengdu",
                    source="从 2025-09-29 起，我现在住在 Chengdu。",
                    observed=timestamp("2025-09-29T12:00:00Z"),
                    start=boundary,
                    end=None,
                )
            ]
        )
        outcome = memory.ingest_drafts(
            [
                draft(
                    value="Ningbo",
                    source="在 2025-09-29 之前，我住在 Ningbo。",
                    observed=timestamp("2025-10-29T12:00:00Z"),
                    start=None,
                    end=boundary,
                )
            ]
        )[0]

        assert outcome.relations[0].relation == expected
    finally:
        memory.close()


def test_temporal_full_recovers_current_and_asof_views(tmp_path) -> None:
    boundary = timestamp("2025-09-24T00:00:00Z")
    memory = build_memory(tmp_path, grounding=True, relation_mode="valid_interval_v2")
    try:
        memory.ingest_drafts(
            [
                draft(
                    value="Tokyo",
                    source="从 2025-09-24 起，我现在住在 Tokyo。",
                    observed=timestamp("2025-09-24T12:00:00Z"),
                    start=boundary,
                    end=None,
                )
            ]
        )
        historical = memory.ingest_drafts(
            [
                draft(
                    value="Suzhou",
                    source="在 2025-09-24 之前，我住在 Suzhou。",
                    observed=timestamp("2025-10-24T12:00:00Z"),
                    start=boundary,
                    end=None,
                )
            ]
        )[0]
        current = memory.search(
            owner_id="u1",
            query="Where do I live now?",
            frame=QueryFrame(
                raw_query="Where do I live now?",
                owner_id="u1",
                target_subject="u1",
                target_predicate="residence.current",
                temporal_intent=TemporalIntent.CURRENT,
                query_time=timestamp("2025-10-24T12:00:00Z"),
                knowledge_time=timestamp("2025-10-25T12:00:00Z"),
                expected_cardinality="one",
            ),
            variant="full",
        )
        past = memory.search(
            owner_id="u1",
            query="Where did I live in August?",
            frame=QueryFrame(
                raw_query="Where did I live in August?",
                owner_id="u1",
                target_subject="u1",
                target_predicate="residence.current",
                temporal_intent=TemporalIntent.AS_OF,
                query_time=timestamp("2025-08-11T12:00:00Z"),
                knowledge_time=timestamp("2025-10-25T12:00:00Z"),
                expected_cardinality="one",
            ),
            variant="full",
        )

        assert historical.relations[0].relation == RelationType.COEXISTS
        assert historical.claim.metadata["temporal_certificate"]["consistency"] == "corrected"
        assert [item.claim.object_value for item in current.claims] == ["Tokyo"]
        assert [item.claim.object_value for item in past.claims] == ["Suzhou"]
    finally:
        memory.close()
