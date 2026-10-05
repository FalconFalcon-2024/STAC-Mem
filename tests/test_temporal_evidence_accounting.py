from __future__ import annotations

import pytest

from stacmem.models import ClaimDraft, QueryFrame
from stacmem.source_spans import SourceSpan
from stacmem.standalone import StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.temporal_evidence import account_temporal_evidence
from stacmem.temporal_grounding import TemporalGrounder
from stacmem.time_utils import ensure_ms


def grounded(source, value="London", predicate="residence.current"):
    draft = ClaimDraft(
        owner_id="u", subject="u", predicate=predicate, object_value=value,
        source_content=source, assertion_time=ensure_ms("2025-10-01"),
        valid_start=ensure_ms("2025-01-01"), valid_end=ensure_ms("2025-12-01"),
    )
    certificate = TemporalGrounder().ground_and_apply(draft, mode="certificate_v2")
    return draft, certificate


@pytest.mark.parametrize("expression", [
    "in Q1 2025", "in spring 2025", "during the summer of 2025",
    "toward the end of 2025", "two months afterward", "a few weeks afterward",
    "in the latter half of 2025", "around mid-2025", "toward Q3",
    "later that quarter", "the next semester", "after the holidays",
    "in Q2", "during autumn", "during winter", "during fall",
    "in the first quarter", "in the second half of the year",
    "at the beginning of the year", "near the middle of the year",
    "towards the end of the year", "several days hence", "three weeks ago",
    "one decade later", "2025", "in 2027", "in 2030",
])
def test_unaccounted_temporal_evidence_cannot_inherit_precise_onset(expression):
    source = f"On 2025-01-01 I joined Aster and moved to London {expression}."
    draft, certificate = grounded(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.binding_scope == "unresolved"
    assert certificate.unresolved_temporal_signal_spans
    assert certificate.temporal_evidence_detector == "temporal-evidence-accounting-v1"
    unresolved = [item for item in certificate.temporal_evidence_accounting
                  if item["disposition"] == "unresolved"]
    assert unresolved
    for item in certificate.temporal_evidence_accounting:
        start, end = item["codepoint_span"]
        assert source[start:end] == item["span"]


@pytest.mark.parametrize("expression", [
    "两个月后", "几个月后", "年底", "年初", "年中", "第一季度", "第二季度",
    "上半年", "下半年", "几周后", "三天以后", "两年之前", "两个月之后",
    "2025年下半年", "2026年春天", "春季", "明年", "次年", "几小时后",
])
def test_chinese_periods_and_relative_units_block_inherited_date(expression):
    source = f"自2025-01-01起我加入Aster，并且{expression}搬到上海。"
    draft, certificate = grounded(source, "上海")
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.consistency == "unsupported"
    assert certificate.binding_scope == "unresolved"
    assert certificate.unresolved_temporal_signal_spans


@pytest.mark.parametrize("expression", ["in Q1 2025", "in spring 2025", "two months afterward"])
def test_evidence_accounting_also_guards_local_cue_and_missed_segmentation(monkeypatch, expression):
    monkeypatch.setattr("stacmem.temporal_grounding.proposition_spans",
                        lambda text: [SourceSpan(text, 0, len(text))])
    draft, certificate = grounded(
        f"On 2025-01-01 I joined Aster and moved to London {expression}."
    )
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.binding_scope == "unresolved"


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster in spring 2025 and moved to London.",
    "On 2025-01-01 I joined Aster and moved to Paris two months afterward and moved to London.",
    "On 2025-01-01, I moved to London in Q1 2025.",
])
def test_anchor_intermediate_event_and_introductory_paths_are_accounted(source):
    draft, certificate = grounded(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.unresolved_temporal_signal_spans


@pytest.mark.parametrize("name", [
    "Summer 2025 Labs", "Spring Labs", "Q1 Labs", "2024", "Next Semester Labs",
    "Afterward Labs", "季度公司", "两个月后公司",
])
def test_literal_names_are_accounted_not_confused_with_temporal_modifiers(name):
    source = f"On 2025-01-01 I joined {name}."
    draft, certificate = grounded(source, name, "employment.organization")
    assert draft.valid_start == ensure_ms("2025-01-01") and draft.valid_end is None
    assert not certificate.unresolved_temporal_signal_spans
    dispositions = {item["disposition"] for item in certificate.temporal_evidence_accounting}
    assert "parsed_cue" in dispositions
    assert "object_literal" in dispositions


@pytest.mark.parametrize("expression", ["after lunch", "after telling my family"])
def test_event_relative_qualifiers_remain_conservatively_unresolved(expression):
    draft, certificate = grounded(
        f"On 2025-01-01 I joined Aster and moved to London {expression}."
    )
    assert draft.valid_start is None and draft.valid_end is None
    assert certificate.unresolved_temporal_signal_spans


@pytest.mark.parametrize("source,value,date,predicate", [
    ("On 2025-01-01 I joined Aster and moved to London.", "London", "2025-01-01",
     "residence.current"),
    ("自2025-01-01起我搬到上海。", "上海", "2025-01-01", "residence.current"),
    ("On 2025-01-01 I joined Aster and on 2025-03-01 I moved to London.",
     "London", "2025-03-01", "residence.current"),
    ("On 2025-01-01 I joined Aster and moved to London by train.",
     "London", "2025-01-01", "residence.current"),
])
def test_supported_time_and_non_temporal_adverbs_still_bind(source, value, date, predicate):
    draft, certificate = grounded(source, value, predicate)
    assert draft.valid_start == ensure_ms(date) and draft.valid_end is None
    assert not certificate.unresolved_temporal_signal_spans
    assert certificate.temporal_evidence_accounting
    assert all(item["disposition"] != "unresolved"
               for item in certificate.temporal_evidence_accounting)


@pytest.mark.parametrize("source,value", [
    ("On 2025-01-01 I joined Aster and moved to London in Q1 2025.", "London"),
    ("自2025-01-01起我加入Aster，并且两个月后搬到上海。", "上海"),
])
def test_unresolved_period_cannot_be_retrieved_before_authenticated_evidence(
    tmp_path, source, value,
):
    path = tmp_path / "memory.db"
    with StandaloneMemory.offline(path) as memory:
        data = fixture("u", "s1", source, value, date="2025-10-01",
                       predicate="residence.current", kind="transition")
        claim = memory.remember(**data)["claims"][0]
        assert claim["status"] == "active" and claim["valid_start"] is None
        assert claim["metadata"]["temporal_eligibility"]["kind"] == "unknown_onset"
    with StandaloneMemory.offline(path) as memory:
        for date, expected in [("2025-01-02", []), ("2025-10-01", [value])]:
            frame = QueryFrame(
                owner_id="u", target_subject="u", target_predicate="residence.current",
                raw_query="Where did I live?", temporal_intent="as_of", query_time=ensure_ms(date),
            )
            pack = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
            assert [row["claim"]["object_value"] for row in pack["claims"]] == expected


@pytest.mark.parametrize("year", ["1998", "2025", "2037", "2101"])
def test_unknown_period_qualifier_cannot_hide_a_bare_year(year):
    draft, certificate = grounded(
        f"On 2025-01-01 I joined Aster and moved to London in the glorp phase of {year}."
    )
    assert draft.valid_start is None and draft.valid_end is None
    assert any(item["kind"] == "bare_year" and item["disposition"] == "unresolved"
               for item in certificate.temporal_evidence_accounting)


@pytest.mark.parametrize("expression,kind", [
    ("Q4", "period"), ("in spring", "period"), ("near year-end", "period"),
    ("two months afterward", "relative_duration"), ("七个月之后", "relative_duration"),
    ("年底", "period"), ("下半年", "period"), ("next semester", "modifier"),
])
def test_compositional_signal_categories_do_not_require_exact_date_parsing(expression, kind):
    source = f"London {expression}"
    accounts = account_temporal_evidence(
        source, SourceSpan(source, 0, len(source)), consumed_cues=[], object_value="London",
    )
    assert any(account.evidence.kind == kind and account.disposition == "unresolved"
               for account in accounts)


def test_same_year_is_not_consumed_by_equal_value_at_another_position():
    source = "On 2025-01-01 I moved to London in Q1 2025"
    cue_end = len("On 2025-01-01")
    accounts = account_temporal_evidence(
        source, SourceSpan(source, 0, len(source)), consumed_cues=[(0, cue_end)],
        object_value="London",
    )
    years = [account for account in accounts if account.evidence.kind == "bare_year"]
    assert [account.disposition for account in years] == ["parsed_cue", "unresolved"]


def test_partial_consumption_or_object_mask_cannot_account_for_whole_expression():
    source = "two months afterward and Spring 2025 Labs"
    accounts = account_temporal_evidence(
        source, SourceSpan(source, 0, len(source)), consumed_cues=[(0, len("two months"))],
        object_value="Labs",
    )
    assert any(account.evidence.kind == "relative_duration" and account.disposition == "unresolved"
               for account in accounts)
    assert any(account.evidence.kind == "bare_year" and account.disposition == "unresolved"
               for account in accounts)


def test_accounting_excludes_embedded_identifiers_and_preserves_unicode_offsets():
    source = "前文。 London during Q1 2025, code X2025B and 123456"
    start = source.index("London")
    accounts = account_temporal_evidence(
        source, SourceSpan(source[start:], start, len(source)), consumed_cues=[],
        object_value="London",
    )
    years = [account.evidence.span for account in accounts if account.evidence.kind == "bare_year"]
    assert years == ["2025"]
    for account in accounts:
        left, right = account.evidence.bounds
        assert source[left:right] == account.evidence.span


def test_accounting_rejects_a_span_not_taken_from_the_original_source():
    with pytest.raises(ValueError, match="original source"):
        account_temporal_evidence(
            "during Q1", SourceSpan("on 2025-01-01", 0, 9),
            consumed_cues=[], object_value="London",
        )
