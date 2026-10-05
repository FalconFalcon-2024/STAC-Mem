from __future__ import annotations

import pytest

from stacmem.models import QueryFrame
from stacmem.retrieval import _temporal_score, _validity_score
from stacmem.standalone import StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.time_utils import ensure_ms


def remember(memory, source, value="Aster", *, session="s1", date="2024-10-01",
             predicate="employment.organization", kind="assertion"):
    data = fixture("u", session, source, value, date=date, predicate=predicate, kind=kind)
    # Unsupported model dates must not override source semantics.
    data["cached_claims"]["claims"][0]["valid_start"] = "1990-01-01"
    return memory.remember(**data)["claims"][0]


def frame(date=None, *, intent="as_of", predicate="employment.organization", **kwargs):
    return QueryFrame(
        owner_id="u", raw_query="What is the state?", target_subject="u",
        target_predicate=predicate, temporal_intent=intent,
        query_time=ensure_ms(date), **kwargs,
    )


def search(memory, query):
    return memory.search(owner_id="u", query=query.raw_query, frame=query)


def values(pack):
    return [c["claim"]["object_value"] for c in pack["claims"]]


@pytest.mark.parametrize("date,expected", [
    ("2000-01-01", []), ("2024-09-30", []),
    ("2024-10-01", ["Aster"]), ("2025-01-01", ["Aster"]),
])
def test_unknown_onset_never_entails_state_before_authenticated_assertion(tmp_path, date, expected):
    path = tmp_path / "memory.db"
    with StandaloneMemory.offline(path) as memory:
        claim = remember(memory, "I work at Aster.")
        assert claim["valid_start"] is None and claim["valid_end"] is None
        eligibility = claim["metadata"]["temporal_eligibility"]
        assert eligibility["kind"] == "unknown_onset" and eligibility["asserts_onset"] is False
        pack = search(memory, frame(date))
        assert values(pack) == expected
        if expected:
            assert "valid=unknown" in pack["context"]
            assert "evidence_from=2024-10-01" in pack["context"]
            assert "not an onset" in pack["context"]
        stored = memory.memory.store.owner_claims("u")[0]
        assert _validity_score(stored, frame(date)) == (1.0 if expected else 0.0)
        assert (_temporal_score(stored, frame(date)) == 1.0) == bool(expected)
    with StandaloneMemory.offline(path) as memory:
        assert values(search(memory, frame(date))) == expected


@pytest.mark.parametrize("start,end,expected", [
    ("2000-01-01", "2024-10-01", []),
    ("2024-09-30", "2024-10-02", ["Aster"]),
])
def test_interval_projection_uses_same_evidence_boundary(tmp_path, start, end, expected):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, "I work at Aster.")
        query = frame(
            intent="interval", interval_start=ensure_ms(start), interval_end=ensure_ms(end)
        )
        assert values(search(memory, query)) == expected


def test_unknown_onset_transition_does_not_backfill_earlier_current_state(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, "I work at Aster.", date="2024-08-01")
        new = remember(memory, "I left Aster and joined Northwind.", "Northwind",
                       date="2024-10-01", session="s2", kind="transition")
        assert new["valid_start"] is None and new["valid_end"] is None
        assert values(search(memory, frame("2024-09-01"))) == ["Aster"]
        assert values(search(memory, frame("2025-01-01", intent="current"))) == ["Northwind"]
        assert values(search(memory, frame("2024-07-31"))) == []


def test_late_bounded_history_and_undated_current_state_do_not_falsely_conflict(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, "I work at Aster.", date="2024-10-01")
        historical = remember(memory, "Before 2024-08-01, I worked at Northwind.", "Northwind",
                              date="2024-11-01", session="s2")
        assert historical["status"] == "active"
        edges = memory.memory.store.relations_for([historical["id"]])
        assert [edge.relation for edge in edges] == ["coexists"]
        assert values(search(memory, frame("2024-07-31"))) == ["Northwind"]
        assert values(search(memory, frame("2024-09-01"))) == []
        assert values(search(memory, frame("2024-12-01"))) == ["Aster"]


@pytest.mark.parametrize("source", [
    "I moved to London, and I work at Aster.",
    "I joined a gym, and I work at Aster.",
    "I left my old apartment, and I work at Aster.",
    "I moved to London and I work at Aster.",
    "I joined a gym and work at Aster.",
])
def test_independent_action_clause_cannot_quarantine_or_supersede_employment(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, "Since 2024-08-01 I work at Northwind.", "Northwind")
        claim = remember(memory, source, session="s2", date="2024-11-01", kind="transition")
        assert claim["status"] == "disputed" and claim["update_kind"] == "assertion"
        certificate = claim["metadata"]["proposition_certificate"]
        assert certificate["support_span"] in {"I work at Aster", "work at Aster"}
        assert certificate["transition_entailment"] is False
        assert not certificate["current_state_reasons"]
        assert certificate["factuality"] == "asserted"
        assert memory.memory.store.relations_for([claim["id"]])[0].relation == "contradicts"


@pytest.mark.parametrize("source", [
    "If I move to London, I work at Aster.",
    "In a hypothetical example, I moved to London and I work at Aster.",
    "I don't, as you know, work at Aster.",
    'My friend said, "I moved to London and I work at Aster."',
])
def test_local_action_scope_still_preserves_outer_nonfactual_scope(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, source)
        assert claim["status"] == "quarantined"
        assert values(search(memory, frame("2025-01-01"))) == []


@pytest.mark.parametrize("source,value,start,end", [
    ("I lived in Paris until 2025-01-01 and then moved to London.", "London", None, None),
    ("I lived in Paris until 2025-01-01 and moved to London.", "London", None, None),
    ("I worked at Northwind until 2025-01-01 and joined Aster.", "Aster", None, None),
    ("Since 2024-08-01 I lived in Paris and on 2025-01-01 I moved to London.",
     "London", "2025-01-01", None),
    ("From 2020-01-01 to 2021-01-01 I worked at Aster, and from 2022-01-01 to "
     "2023-01-01 I worked at Northwind.", "Northwind", "2022-01-01", "2023-01-01"),
    ("From 2020-01-01 to 2021-01-01 I worked at Aster and from 2022-01-01 to "
     "2023-01-01 I worked at Northwind.", "Northwind", "2022-01-01", "2023-01-01"),
    ("Since 2024-08-01 I live in Paris and on 2025-01-01 I moved to London.",
     "Paris", "2024-08-01", None),
])
def test_date_attaches_only_to_corresponding_proposition(tmp_path, source, value, start, end):
    predicate = "residence.current" if value in {"Paris", "London"} else "employment.organization"
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, source, value, predicate=predicate, date="2025-02-01")
        assert claim["valid_start"] == ensure_ms(start) and claim["valid_end"] == ensure_ms(end)
        assert claim["status"] == "active"
        if start:
            certificate = claim["metadata"]["temporal_certificate"]
            left, right = certificate["temporal_cue_codepoint_span"]
            assert source[left:right] == certificate["temporal_cue_span"]
            left, right = certificate["value_support_codepoint_span"]
            assert source[left:right] == certificate["value_support_span"]


def test_two_historical_intervals_are_queryable_without_mutating_each_other(tmp_path):
    source = ("From 2020-01-01 to 2021-01-01 I worked at Aster, and from 2022-01-01 to "
              "2023-01-01 I worked at Northwind.")
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, source, "Aster", date="2024-01-01")
        remember(memory, source, "Northwind", session="s2", date="2024-01-01")
        assert values(search(memory, frame("2020-06-01"))) == ["Aster"]
        assert values(search(memory, frame("2021-06-01"))) == []
        assert values(search(memory, frame("2022-06-01"))) == ["Northwind"]
        assert values(search(memory, frame("2024-01-01"))) == []


@pytest.mark.parametrize("source", [
    "On 2024-08-01 I worked at Aster.",
    "On 2024-08-01, I worked at Aster.",
    "As of 2024-08-01, I still work at Aster.",
])
def test_explicit_historical_day_is_observation_not_onset_or_termination(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, source, kind="transition")
        assert claim["status"] == "active" and claim["update_kind"] == "assertion"
        observation = claim["metadata"]["temporal_observation"]
        assert observation["kind"] == "holds_at"
        assert not observation["asserts_onset"] and not observation["asserts_termination"]
        for date, expected in [("2024-07-31", []), ("2024-08-01T18:00:00Z", ["Aster"]),
                               ("2024-08-02", []), ("2025-01-01", [])]:
            assert values(search(memory, frame(date))) == expected


def test_historical_observation_is_not_retroactive_conflict_with_unknown_current(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, "I work at Aster.", date="2024-10-01")
        old = remember(memory, "On 2024-08-01 I worked at Northwind.", "Northwind", session="s2")
        assert old["status"] == "active"
        assert memory.memory.store.relations_for([old["id"]])[0].relation == "coexists"
        assert values(search(memory, frame("2024-08-01"))) == ["Northwind"]
        assert values(search(memory, frame("2024-10-01"))) == ["Aster"]


@pytest.mark.parametrize("source", [
    "On 2024-08-01 I worked at Northwind, and I work at Aster.",
    "As of 2024-08-01, I still work at Northwind, and I work at Aster.",
    "Since 2024-08-01 I work at Northwind, and I work at Aster.",
])
def test_observation_or_onset_of_other_proposition_is_not_inherited(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, source)
        assert claim["status"] == "active"
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert "temporal_observation" not in claim["metadata"]
        assert values(search(memory, frame("2024-08-01"))) == []
        assert values(search(memory, frame("2024-10-01"))) == ["Aster"]


def test_multiple_local_cues_do_not_guess_an_interval(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, "Since 2020-01-01 until 2021-01-01 I worked at Aster.")
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert claim["metadata"]["temporal_certificate"]["consistency"] == "ambiguous"
        assert claim["status"] == "quarantined"


@pytest.mark.parametrize("prefix", ["On 2024-10-01", "Since 2024-10-01", "On 2024-10-01,"])
def test_shared_departure_arrival_boundary_is_preserved(tmp_path, prefix):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        remember(memory, "Since 2024-08-01 I work at Northwind.", "Northwind",
                 date="2024-08-01")
        source = f"{prefix} I left Northwind and joined Aster."
        claim = remember(memory, source, session="s2", kind="transition")
        assert claim["status"] == "active"
        assert claim["valid_start"] == ensure_ms("2024-10-01") and claim["valid_end"] is None
        assert values(search(memory, frame("2024-09-01"))) == ["Northwind"]
        assert values(search(memory, frame("2025-01-01"))) == ["Aster"]


def test_identical_dates_in_distinct_propositions_are_not_deduplicated(tmp_path):
    source = "On 2024-08-01 I worked at Northwind; on 2024-08-01 I worked at Aster."
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, source)
        assert claim["status"] == "active"
        assert claim["valid_start"] == ensure_ms("2024-08-01")
        assert claim["valid_end"] == ensure_ms("2024-08-02")
        cert = claim["metadata"]["temporal_certificate"]
        assert cert["value_support_span"] == "on 2024-08-01 I worked at Aster"


def test_repeated_value_with_distinct_dates_does_not_select_first_mention(tmp_path):
    source = ("From 2020-01-01 to 2021-01-01 I worked at Aster; "
              "from 2022-01-01 to 2023-01-01 I worked at Aster.")
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = remember(memory, source)
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert claim["metadata"]["temporal_certificate"]["object_grounded"] is False
        assert claim["status"] == "quarantined"
