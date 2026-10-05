from __future__ import annotations

import pytest

from stacmem.models import ClaimDraft, QueryFrame
from stacmem.source_spans import SourceSpan, proposition_spans
from stacmem.standalone import StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.temporal_grounding import TemporalGrounder
from stacmem.time_utils import ensure_ms


def ground(source, value="London", *, predicate="residence.current"):
    draft = ClaimDraft(
        owner_id="u", subject="u", predicate=predicate, object_value=value,
        source_content=source, assertion_time=ensure_ms("2025-04-01"),
        valid_start=ensure_ms("2025-01-01"), valid_end=ensure_ms("2025-05-01"),
    )
    certificate = TemporalGrounder().ground_and_apply(draft, mode="certificate_v2")
    return draft, certificate


@pytest.mark.parametrize("target", [
    "moved to London by 2025-03-01",
    "moved to London in 2025-03-01",
    "by 2025-03-01 moved to London",
    "moved to London after 2025-03-01",
    "moved to London around 2025-03-01",
    "moved to London during 2025-03-01",
    "moved to London in 2025-03",
    "moved to London by 2025/03/01",
    "moved to London in March 2025",
    "moved to London by March 1, 2025",
    "moved to London 2025-03-01",
    "moved to London by 2025-01-01",
])
def test_unparsed_target_date_vetoes_shared_date_even_when_the_date_is_equal(target):
    source = f"On 2025-01-01 I joined Aster and {target}."
    draft, certificate = ground(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.binding_scope == "unresolved"
    assert "independent_temporal_signal_blocks_binding" in certificate.reasons
    assert certificate.unresolved_temporal_signal_spans
    for text, (start, end) in zip(
        certificate.unresolved_temporal_signal_spans,
        certificate.unresolved_temporal_signal_codepoint_spans, strict=True,
    ):
        assert source[start:end] == text


@pytest.mark.parametrize("target", [
    "eventually moved to London", "soon after moved to London",
    "shortly after moved to London", "the following month moved to London",
    "following month moved to London", "moved to London eventually",
    "moved to London soon after", "moved to London shortly after",
    "moved to London the following month", "later moved to London",
    "subsequently moved to London", "next week moved to London",
    "after graduation moved to London", "moved to London sometime",
])
def test_independent_temporal_modifier_vetoes_parent_date(target):
    draft, certificate = ground(f"On 2025-01-01 I joined Aster and {target}.")
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.binding_scope == "unresolved"
    assert certificate.unresolved_temporal_signal_spans


@pytest.mark.parametrize("target", [
    "moved to London by 2025-03-01", "eventually moved to London",
    "shortly after moved to London", "the following month moved to London",
])
def test_veto_survives_missed_proposition_segmentation(monkeypatch, target):
    monkeypatch.setattr("stacmem.temporal_grounding.proposition_spans",
                        lambda text: [SourceSpan(text, 0, len(text))])
    draft, certificate = ground(f"On 2025-01-01 I joined Aster and {target}.")
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.binding_scope == "unresolved"


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster by 2025-03-01 and moved to London.",
    "On 2025-01-01 I joined Aster and moved to Paris by 2025-03-01 and moved to London.",
    "On 2025-01-01 I joined Aster and eventually moved to Paris and moved to London.",
    "On 2025-01-01, I moved to London by 2025-03-01.",
    "On 2025-01-01, eventually I moved to London.",
])
def test_unparsed_temporal_signal_stops_the_entire_inheritance_path(source):
    draft, certificate = ground(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.binding_scope == "unresolved"


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I moved to London by 2025-03-01.",
    "On 2025-01-01 I eventually moved to London.",
    "Since 2025-01-01 I moved to London the following month.",
])
def test_supported_local_cue_does_not_hide_unparsed_local_evidence(source):
    draft, certificate = ground(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"


@pytest.mark.parametrize("source,value,predicate", [
    ("On 2025-01-01 I joined Aster and moved to London.", "London", "residence.current"),
    ("On 2025-01-01, I moved to London.", "London", "residence.current"),
    ("Since 2025-01-01 I moved to London.", "London", "residence.current"),
    ("On 2025-01-01 I joined Eventually Labs.", "Eventually Labs", "employment.organization"),
    ("On 2025-01-01 I joined March 2025 Labs.", "March 2025 Labs", "employment.organization"),
    ("On 2025-01-01 I moved to London and joined Next Month Labs.",
     "Next Month Labs", "employment.organization"),
    ("On 2025-01-01 I moved to London and joined Eventually Labs.",
     "Eventually Labs", "employment.organization"),
    ("On 2025-01-01 I joined Last Works and moved to London.", "London", "residence.current"),
    ("On 2025-01-01 I joined Aster and on 2025-03-01 I moved to London.",
     "London", "residence.current"),
])
def test_resolved_cues_and_literal_object_names_are_not_unparsed_signals(source, value, predicate):
    draft, certificate = ground(source, value, predicate=predicate)
    expected = "2025-03-01" if "on 2025-03-01" in source else "2025-01-01"
    assert draft.valid_start == ensure_ms(expected)
    assert draft.valid_end is None
    assert certificate.consistency == "corrected"
    assert not certificate.unresolved_temporal_signal_spans


def test_supported_closed_interval_consumes_both_date_tokens():
    draft, certificate = ground("I lived in London from 2024-01-01 until 2025-01-01.")
    assert draft.valid_start == ensure_ms("2024-01-01")
    assert draft.valid_end == ensure_ms("2025-01-01")
    assert certificate.direction == "closed_interval"
    assert not certificate.unresolved_temporal_signal_spans


@pytest.mark.parametrize("source", [
    "On 2025-02-30 I moved to London.",
    "Since 2025-13-01 I moved to London.",
    "I lived in London from 2025-02-30 until 2025-03-01.",
    "On 2025-01-01 I joined Aster and moved to London by 2025-02-30.",
])
def test_invalid_calendar_dates_remain_unparsed_evidence_not_binding_authority(source):
    draft, certificate = ground(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.unresolved_temporal_signal_spans


def test_first_dated_event_is_not_changed_by_independent_next_clause_date():
    source = "On 2025-01-01 I joined Aster and moved to London by 2025-03-01."
    first, certificate = ground(source, "Aster", predicate="employment.organization")
    assert first.valid_start == ensure_ms("2025-01-01")
    assert certificate.binding_scope == "local"


def test_unknown_onset_cannot_answer_retrospective_query_and_survives_reopen(tmp_path):
    path = tmp_path / "memory.db"
    source = "On 2025-01-01 I joined Aster and moved to London by 2025-03-01."
    with StandaloneMemory.offline(path) as memory:
        data = fixture("u", "s1", source, "London", date="2025-04-01",
                       predicate="residence.current", kind="transition")
        claim = memory.remember(**data)["claims"][0]
        assert claim["status"] == "active"
        assert claim["valid_start"] is None and claim["valid_end"] is None
        assert claim["metadata"]["temporal_eligibility"]["kind"] == "unknown_onset"
    with StandaloneMemory.offline(path) as memory:
        for date, expected in [("2025-01-01", []), ("2025-04-01", ["London"])]:
            frame = QueryFrame(
                owner_id="u", raw_query="Where did I live?", target_subject="u",
                target_predicate="residence.current", temporal_intent="as_of",
                query_time=ensure_ms(date),
            )
            pack = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
            assert [row["claim"]["object_value"] for row in pack["claims"]] == expected


@pytest.mark.parametrize("connector", ["and then at", "then at", "and then in", "then in"])
def test_sequential_historical_ellipsis_binds_own_day_and_inherits_past_tense(tmp_path, connector):
    source = f"I worked at Aster on 2024-08-01 {connector} Northwind on 2025-01-01."
    spans = proposition_spans(source)
    assert len(spans) == 2
    for span in spans:
        assert source[span.start:span.end] == span.text
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        for session, value, date in [("s1", "Aster", "2024-08-01"),
                                     ("s2", "Northwind", "2025-01-01")]:
            data = fixture("u", session, source, value, date="2025-04-01",
                           predicate="employment.organization", kind="assertion")
            claim = memory.remember(**data)["claims"][0]
            assert claim["valid_start"] == ensure_ms(date)
            assert claim["status"] == "active"
            assert claim["metadata"]["temporal_observation"]["asserts_onset"] is False
        for date, expected in [("2024-08-01", ["Aster"]), ("2025-01-01", ["Northwind"]),
                               ("2025-01-02", [])]:
            frame = QueryFrame(
                owner_id="u", raw_query="Where did I work?", target_subject="u",
                target_predicate="employment.organization", temporal_intent="as_of",
                query_time=ensure_ms(date),
            )
            pack = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
            assert [row["claim"]["object_value"] for row in pack["claims"]] == expected


def test_undated_sequential_past_ellipsis_stays_quarantined(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        data = fixture("u", "s1", "I worked at Aster and then at Northwind.", "Northwind",
                       date="2025-04-01", predicate="employment.organization", kind="assertion")
        claim = memory.remember(**data)["claims"][0]
        assert claim["status"] == "quarantined"
        assert "past_only_state_without_supported_end" in claim["metadata"]["admission_reasons"]


def test_departure_postfix_date_does_not_become_arrival_onset():
    draft, certificate = ground("I left Northwind on 2025-01-01 and joined Aster.", "Aster",
                                predicate="employment.organization")
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.binding_scope == "unresolved"
