from __future__ import annotations

import pytest

from stacmem.models import ClaimDraft, QueryFrame
from stacmem.source_spans import SourceSpan, proposition_spans
from stacmem.standalone import StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.temporal_grounding import TemporalGrounder
from stacmem.time_utils import ensure_ms


def write(memory, source, value, *, predicate="residence.current", session="s1"):
    data = fixture("u", session, source, value, date="2025-02-01", predicate=predicate,
                   kind="transition")
    # Deliberately give the model an incorrect endpoint, exercising source authority.
    data["cached_claims"]["claims"][0]["valid_start"] = None
    data["cached_claims"]["claims"][0]["valid_end"] = "2025-01-01"
    return memory.remember(**data)["claims"][0]


def query(memory, date, *, predicate="residence.current"):
    frame = QueryFrame(
        owner_id="u", target_subject="u", target_predicate=predicate,
        raw_query="What was the state?", temporal_intent="as_of", query_time=ensure_ms(date),
    )
    pack = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
    return [row["claim"]["object_value"] for row in pack["claims"]]


@pytest.mark.parametrize("connector,action", [
    ("then", "moved to London"), ("before", "moving to London"),
    ("after", "moving to London"), ("prior to", "moving to London"),
    ("then I", "moved to London"), ("and then", "moved to London"),
])
def test_old_residence_endpoint_cannot_be_new_residence_endpoint(tmp_path, connector, action):
    source = f"I lived in Paris until 2025-01-01 {connector} {action}."
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert claim["status"] == "active"
        assert query(memory, "2024-12-31") == []
        assert query(memory, "2025-01-01") == []
        assert query(memory, "2025-02-01") == ["London"]
        assert query(memory, "2026-01-01") == ["London"]


@pytest.mark.parametrize("connector,action", [
    ("then", "joined Aster"), ("before", "joining Aster"),
    ("after", "joining Aster"), ("prior to", "joining Aster"),
    ("then I", "joined Aster"),
])
def test_old_employment_endpoint_cannot_be_new_employment_endpoint(tmp_path, connector, action):
    source = f"I worked at Northwind until 2025-01-01 {connector} {action}."
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "Aster", predicate="employment.organization")
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert claim["status"] == "active"
        assert query(memory, "2024-12-31", predicate="employment.organization") == []
        assert query(memory, "2025-02-01", predicate="employment.organization") == ["Aster"]


@pytest.mark.parametrize("source,value,predicate", [
    ("I lived in Paris until 2025-01-01 upon moving to London.", "London", "residence.current"),
    ("I worked at Northwind until 2025-01-01 subsequently joined Aster.",
     "Aster", "employment.organization"),
    ("I moved to London before 2025-01-01.", "London", "residence.current"),
    ("I joined Aster before 2025-01-01.", "Aster", "employment.organization"),
])
def test_arrival_event_upper_bound_fails_closed_even_without_known_connector(
    tmp_path, source, value, predicate
):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, value, predicate=predicate)
        cert = claim["metadata"]["temporal_certificate"]
        assert cert["consistency"] == "unsupported"
        assert "new_value_upper_bound_requires_independent_endpoint_evidence" in cert["reasons"]
        assert cert["parsed_end"] == ensure_ms("2025-01-01")
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert query(memory, "2024-12-31", predicate=predicate) == []


def test_endpoint_guard_is_independent_of_proposition_segmentation(monkeypatch):
    source = "I lived in Paris until 2025-01-01 then moved to London."
    monkeypatch.setattr("stacmem.temporal_grounding.proposition_spans",
                        lambda text: [SourceSpan(text, 0, len(text))])
    draft = ClaimDraft(
        owner_id="u", subject="u", predicate="residence.current", object_value="London",
        source_content=source, assertion_time=ensure_ms("2025-02-01"),
        valid_end=ensure_ms("2025-01-01"),
    )
    cert = TemporalGrounder().ground_and_apply(draft, mode="certificate_v2")
    assert cert.consistency == "unsupported"
    assert cert.reasons == ("new_value_upper_bound_requires_independent_endpoint_evidence",)
    assert draft.valid_start is None and draft.valid_end is None


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster and moved to London.",
    "On 2025-01-01 I moved to London and joined Aster.",
    "On 2025-01-01 I moved to London and started working at Aster.",
    "On 2025-01-01, I joined Aster and moved to London.",
    "On 2025-01-01 I joined Aster and I moved to London.",
    "On 2025-01-01 I joined Aster then moved to London.",
])
def test_introductory_date_covers_both_coordinated_events_and_survives_reopen(tmp_path, source):
    path = tmp_path / "memory.db"
    with StandaloneMemory.offline(path) as memory:
        claims = [
            write(memory, source, "London", session="residence"),
            write(memory, source, "Aster", predicate="employment.organization", session="job"),
        ]
        for claim in claims:
            assert claim["valid_start"] == ensure_ms("2025-01-01")
            assert claim["valid_end"] is None and claim["status"] == "active"
            cert = claim["metadata"]["temporal_certificate"]
            left, right = cert["binding_scope_codepoint_span"]
            assert source[left:right].startswith("On 2025-01-01")
        assert any(c["metadata"]["temporal_certificate"]["binding_scope"] == "coordinated_events"
                   for c in claims)
    with StandaloneMemory.offline(path) as memory:
        assert query(memory, "2024-12-31") == []
        assert query(memory, "2025-01-01") == ["London"]
        assert query(memory, "2026-01-01") == ["London"]
        assert query(memory, "2025-01-01", predicate="employment.organization") == ["Aster"]


def test_three_connected_events_share_one_introductory_date(tmp_path):
    source = "On 2025-01-01 I left Northwind and joined Aster and moved to London."
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["valid_start"] == ensure_ms("2025-01-01")
        assert claim["metadata"]["temporal_certificate"]["binding_scope"] == "coordinated_events"


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster and on 2025-03-01 I moved to London.",
    "On 2025-01-01 I joined Aster then on 2025-03-01 I moved to London.",
    "On 2025-01-01 I joined Aster, but on 2025-03-01 I moved to London.",
])
def test_new_local_date_replaces_shared_scope(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["valid_start"] == ensure_ms("2025-03-01")
        assert claim["metadata"]["temporal_certificate"]["binding_scope"] == "local"
        assert query(memory, "2025-01-01") == []
        assert query(memory, "2025-03-01") == ["London"]


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster, but I moved to London.",
    "On 2025-01-01 I joined Aster. I moved to London.",
    "On 2025-01-01 I joined Aster; I moved to London.",
    "On 2025-01-01 I joined Aster before moving to London.",
    "On 2025-01-01 I joined Aster and later moved to London.",
    "On 2025-01-01 I joined Aster subsequently moved to London.",
    "I joined Aster on 2025-01-01 and moved to London.",
    "Since 2025-01-01 I joined Aster then moved to London.",
])
def test_scope_does_not_cross_barriers_postfix_dates_or_independent_time(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert query(memory, "2025-01-01") == []


@pytest.mark.parametrize("source", [
    "I worked at Aster on 2024-08-01 and at Northwind on 2025-01-01.",
    "On 2024-08-01 I worked at Aster then on 2025-01-01 I worked at Northwind.",
    "I worked at Aster on 2024-08-01 then on 2025-01-01 I worked at Northwind.",
])
def test_coordinated_historical_observations_bind_their_own_dates(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        old = write(memory, source, "Aster", predicate="employment.organization", session="old")
        new = write(memory, source, "Northwind", predicate="employment.organization", session="new")
        assert old["valid_start"] == ensure_ms("2024-08-01")
        assert new["valid_start"] == ensure_ms("2025-01-01")
        for claim in (old, new):
            assert claim["status"] == "active"
            assert claim["metadata"]["temporal_observation"]["kind"] == "holds_at"
        assert query(memory, "2024-08-01", predicate="employment.organization") == ["Aster"]
        assert query(memory, "2025-01-01", predicate="employment.organization") == ["Northwind"]
        assert query(memory, "2025-01-02", predicate="employment.organization") == []


def test_undated_past_ellipsis_cannot_become_current(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, "I worked at Aster and at Northwind.", "Northwind",
                      predicate="employment.organization")
        assert claim["status"] == "quarantined"
        assert "past_only_state_without_supported_end" in claim["metadata"]["admission_reasons"]


@pytest.mark.parametrize("source", [
    "I lived in London until 2025-01-01.", "Until 2025-01-01 I lived in London.",
])
def test_independent_historical_endpoint_remains_valid(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["status"] == "active"
        assert claim["valid_end"] == ensure_ms("2025-01-01")
        assert query(memory, "2024-12-31") == ["London"]
        assert query(memory, "2025-01-01") == []


def test_temporal_scope_does_not_treat_last_in_a_company_name_as_a_relative_date(tmp_path):
    source = "On 2025-01-01 I joined Last Works and moved to London."
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["valid_start"] == ensure_ms("2025-01-01")


@pytest.mark.parametrize("source", [
    "If I win, on 2025-01-01 I joined Aster and moved to London.",
    "In a fictional example, on 2025-01-01 I joined Aster and moved to London.",
    'My friend said, "On 2025-01-01 I joined Aster and moved to London."',
])
def test_shared_time_scope_does_not_weaken_source_factuality(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = write(memory, source, "London")
        assert claim["status"] == "quarantined"
        assert query(memory, "2025-01-01") == []


def test_transition_boundaries_preserve_literal_value_offsets():
    source = "I lived in Paris until 2025-01-01 before moving to London."
    spans = proposition_spans(source)
    assert [span.text for span in spans] == [
        "I lived in Paris until 2025-01-01", "moving to London",
    ]
    for span in spans:
        assert source[span.start:span.end] == span.text
