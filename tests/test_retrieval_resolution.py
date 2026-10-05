from __future__ import annotations

from stacmem.models import ClaimDraft, QueryFrame, TemporalIntent, UpdateKind


def add(
    memory,
    value: str,
    observed: int,
    start: int,
    kind: UpdateKind,
    predicate: str = "residence.current",
) -> None:
    memory.ingest_drafts(
        [
            ClaimDraft(
                owner_id="u1",
                subject="Alice",
                predicate=predicate,
                object_value=value,
                assertion_time=observed,
                observed_at=observed,
                valid_start=start,
                update_kind=kind,
                functional=True,
                source_content=f"Alice lives in {value}",
            )
        ]
    )


def frame(intent: TemporalIntent, query_time: int, knowledge_time: int = 10_000) -> QueryFrame:
    return QueryFrame(
        raw_query="Where does Alice live?",
        owner_id="u1",
        target_subject="Alice",
        target_predicate="residence.current",
        temporal_intent=intent,
        query_time=query_time,
        knowledge_time=knowledge_time,
    )


def test_current_and_as_of_projection(memory) -> None:
    add(memory, "City A", observed=1000, start=1000, kind=UpdateKind.ASSERTION)
    add(memory, "City B", observed=2000, start=2000, kind=UpdateKind.TRANSITION)

    current = memory.search(
        owner_id="u1",
        query="Where does Alice currently live?",
        frame=frame(TemporalIntent.CURRENT, 3000),
        variant="full",
    )
    assert [item.claim.object_value for item in current.claims] == ["City B"]

    historical = memory.search(
        owner_id="u1",
        query="Where did Alice live before?",
        frame=frame(TemporalIntent.AS_OF, 1500),
        variant="full",
    )
    assert [item.claim.object_value for item in historical.claims] == ["City A"]


def test_known_as_of_before_correction(memory) -> None:
    add(
        memory,
        "Wrong Co",
        observed=1000,
        start=500,
        kind=UpdateKind.ASSERTION,
        predicate="employment.organization",
    )
    add(
        memory,
        "Right Co",
        observed=2000,
        start=500,
        kind=UpdateKind.CORRECTION,
        predicate="employment.organization",
    )
    query = QueryFrame(
        raw_query="Where did Alice work?",
        owner_id="u1",
        target_subject="Alice",
        target_predicate="employment.organization",
        temporal_intent=TemporalIntent.AS_OF,
        query_time=700,
        knowledge_time=1500,
    )
    result = memory.search(
        owner_id="u1",
        query=query.raw_query,
        frame=query,
        variant="full",
    )
    assert [item.claim.object_value for item in result.claims] == ["Wrong Co"]


def test_unresolved_overlap_is_exposed_instead_of_silently_resolved(memory) -> None:
    add(memory, "City A", observed=1000, start=500, kind=UpdateKind.ASSERTION)
    add(memory, "City B", observed=2000, start=500, kind=UpdateKind.ASSERTION)
    result = memory.search(
        owner_id="u1",
        query="Where does Alice live?",
        frame=frame(TemporalIntent.CURRENT, 3000),
        variant="full",
    )
    assert {item.claim.object_value for item in result.claims} == {"City A", "City B"}
    assert result.diagnostics["unresolved_slots"] == 1
    assert "do not silently choose" in result.context


def test_query_predicate_alias_is_canonicalized(memory) -> None:
    add(
        memory,
        "Northwind Labs",
        observed=1000,
        start=500,
        kind=UpdateKind.ASSERTION,
        predicate="works_at",
    )
    query = QueryFrame(
        raw_query="Where does Alice work?",
        owner_id="u1",
        target_subject="Alice",
        target_predicate="job.employer",
        temporal_intent=TemporalIntent.CURRENT,
        query_time=2000,
    )
    result = memory.search(
        owner_id="u1",
        query=query.raw_query,
        frame=query,
        variant="full",
    )
    assert query.target_predicate == "employment.organization"
    assert [item.claim.object_value for item in result.claims] == ["Northwind Labs"]


def test_explicit_target_slot_excludes_unrelated_predicates(memory) -> None:
    add(
        memory,
        "Northwind Labs",
        observed=1000,
        start=500,
        kind=UpdateKind.ASSERTION,
        predicate="employment.current",
    )
    add(
        memory,
        "product designer",
        observed=1000,
        start=500,
        kind=UpdateKind.ASSERTION,
        predicate="occupation.current",
    )
    query = QueryFrame(
        raw_query="Where does Alice currently work?",
        owner_id="u1",
        target_subject="Alice",
        target_predicate="employment.current",
        temporal_intent=TemporalIntent.CURRENT,
        query_time=2000,
    )
    result = memory.search(
        owner_id="u1",
        query=query.raw_query,
        frame=query,
        variant="full",
    )

    assert [item.claim.object_value for item in result.claims] == ["Northwind Labs"]
    assert result.diagnostics["input_candidates"] == 1
    assert result.diagnostics["query_routing"] == "structured_slot"


def test_qwen_like_employment_change_supports_current_asof_and_history(memory) -> None:
    memory.ingest_drafts(
        [
            ClaimDraft(
                owner_id="u1",
                subject="Alice",
                predicate="employment.current",
                object_value="Northwind Labs",
                assertion_time=1000,
                observed_at=1000,
                valid_start=500,
                update_kind=UpdateKind.ASSERTION,
                functional=True,
                source_content="I started working at Northwind Labs.",
            ),
            ClaimDraft(
                owner_id="u1",
                subject="Alice",
                predicate="occupation.current",
                object_value="product designer",
                assertion_time=1000,
                observed_at=1000,
                valid_start=500,
                update_kind=UpdateKind.ASSERTION,
                functional=True,
                source_content="I started working as a product designer.",
            ),
        ]
    )
    memory.ingest_drafts(
        [
            ClaimDraft(
                owner_id="u1",
                subject="Alice",
                predicate="employment.current",
                object_value="Contoso Health",
                assertion_time=2000,
                observed_at=2000,
                valid_start=1500,
                update_kind=UpdateKind.ASSERTION,
                functional=True,
                source_content="I left Northwind Labs and joined Contoso Health.",
            ),
            ClaimDraft(
                owner_id="u1",
                subject="Alice",
                predicate="employment.position",
                object_value="senior product designer",
                assertion_time=2000,
                observed_at=2000,
                valid_start=1500,
                update_kind=UpdateKind.ASSERTION,
                functional=True,
                source_content="I joined Contoso Health as a senior product designer.",
            ),
        ]
    )

    def search(intent: TemporalIntent, when: int | None, expected: str = "one"):
        query = QueryFrame(
            raw_query="Where does Alice work?",
            owner_id="u1",
            target_subject="Alice",
            target_predicate="employment.current",
            temporal_intent=intent,
            query_time=when,
            expected_cardinality=expected,
        )
        return memory.search(
            owner_id="u1",
            query=query.raw_query,
            frame=query,
            variant="full",
        )

    current = search(TemporalIntent.CURRENT, 3000)
    as_of = search(TemporalIntent.AS_OF, 1000)
    history = search(TemporalIntent.HISTORY, None, "many")

    assert [item.claim.object_value for item in current.claims] == ["Contoso Health"]
    assert [item.claim.object_value for item in as_of.claims] == ["Northwind Labs"]
    assert {item.claim.object_value for item in history.claims} == {
        "Northwind Labs",
        "Contoso Health",
    }
    assert all(item.claim.predicate == "employment.organization" for item in history.claims)


def test_quarantined_claim_is_excluded_from_every_retrieval_variant(memory) -> None:
    add(
        memory,
        "Northwind Labs",
        observed=1000,
        start=500,
        kind=UpdateKind.ASSERTION,
        predicate="works_at",
    )
    unsafe = ClaimDraft(
        owner_id="u1",
        subject="Alice",
        predicate="works_at",
        object_value="Contoso Health",
        assertion_time=2000,
        observed_at=2000,
        valid_start=1500,
        update_kind=UpdateKind.ASSERTION,
        functional=True,
        source_content="If I joined Contoso Health, I would leave Northwind Labs.",
    )
    memory.ingest_drafts([unsafe])
    query = QueryFrame(
        raw_query="Where does Alice work?",
        owner_id="u1",
        target_subject="Alice",
        target_predicate="works_at",
        temporal_intent=TemporalIntent.CURRENT,
        query_time=3000,
    )

    for variant in ("semantic", "full"):
        result = memory.search(
            owner_id="u1",
            query=query.raw_query,
            frame=query.model_copy(deep=True),
            variant=variant,
        )
        assert [item.claim.object_value for item in result.claims] == ["Northwind Labs"]
