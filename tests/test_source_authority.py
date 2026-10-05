from __future__ import annotations

import pytest

from stacmem.models import QueryFrame
from stacmem.standalone import StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.time_utils import ensure_ms


def proposal(source, *, quote=None, evidence=None, kind="transition", hints=None):
    data = fixture("u", "attempt", source, "Aster", date="2024-10-01", kind=kind)
    row = data["cached_claims"]["claims"][0]
    row["source_content"] = source if quote is None else quote
    row["proposition_grounding"].update(hints or {})
    if evidence is not None:
        row["proposition_grounding"]["evidence_span"] = evidence
    return data


@pytest.mark.parametrize("source,quote,evidence", [
    ("My friend works at Aster.", None, "Aster"),
    ("I might work at Aster next year.", None, "Aster"),
    ("Do I work at Aster?", "I work at Aster", "I work at Aster"),
    ("I work at Aster?", "I work at Aster", "I work at Aster"),
    ("In a hypothetical example, I work at Aster.", "I work at Aster.", "Aster"),
    ("In a screenplay, I work at Aster.", "I work at Aster.", "I work at Aster."),
    ("If I win the lottery, I work at Aster.", "I work at Aster.", "Aster"),
    ("I don't work at Aster.", None, "Aster"),
    ("I don't, as you know, work at Aster.", "work at Aster.", "Aster"),
    ('My friend said, "I work at Aster."', "I work at Aster.", "I work at Aster."),
    ("我的朋友在 Aster 工作。", None, "Aster"),
    ("我可能在 Aster 工作。", None, "Aster"),
    ("我在 Aster 工作吗？", "我在 Aster 工作", "Aster"),
    ("在假设练习中，我在 Aster 工作。", "我在 Aster 工作。", "Aster"),
])
def test_model_hints_and_cropped_quotes_cannot_replace_existing_user_state(
    tmp_path, source, quote, evidence
):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        memory.remember(**fixture("u", "initial", "On 2024-08-01 I joined Northwind.",
                                  "Northwind"))
        result = memory.remember(**proposal(source, quote=quote, evidence=evidence))
        claim = result["claims"][0]
        assert claim["status"] == "quarantined"
        assert claim["source_content"] == source
        assert claim["metadata"]["source_contract"]["status"] == "validated"
        assert claim["metadata"]["proposition_certificate"]["transition_entailment"] is False
        assert memory.memory.store.stats()["counts"]["conflict_relations"] == 0
        old = next(c for c in memory.memory.store.owner_claims("u")
                   if c.object_value == "Northwind")
        assert old.object_value == "Northwind" and old.status == "active"
        assert old.valid_end is None
        frame = QueryFrame(raw_query="Where do I work?", owner_id="u", target_subject="u",
                           target_predicate="employment.organization", temporal_intent="current",
                           query_time=ensure_ms("2025-01-01"))
        resolved = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
        assert [item["claim"]["object_value"] for item in resolved["claims"]] == ["Northwind"]
        assert memory.source_search(owner_id="u", query="Aster")


@pytest.mark.parametrize("source", [
    "I work at Aster.",
    'My employer is "Aster".',
    "I might sound indecisive, yet I joined Aster.",
    "I don't work at Northwind now; I joined Aster.",
    "我在 Aster 工作。",
])
def test_independent_assertions_remain_usable_with_short_evidence(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        result = memory.remember(**proposal(source, evidence="Aster"))
        claim = result["claims"][0]
        assert claim["status"] == "active"
        cert = claim["metadata"]["proposition_certificate"]
        assert cert["factuality"] == "asserted" and cert["subject_alignment"] == "aligned"
        left, right = cert["support_codepoint_span"]
        assert source[left:right] == cert["support_span"]


@pytest.mark.parametrize("hints", [
    {"factuality": "nonfactual"}, {"factuality": "unsupported"},
    {"subject_alignment": "misaligned"}, {"subject_alignment": "unknown"},
])
def test_semantic_hints_can_only_reduce_authority(tmp_path, hints):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        result = memory.remember(**proposal("I work at Aster.", hints=hints))
        assert result["claims"][0]["status"] == "quarantined"


@pytest.mark.parametrize("kind", ["assertion", "transition"])
def test_transition_hint_and_label_without_evidence_preserve_unresolved_conflict(tmp_path, kind):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        memory.remember(**fixture("u", "initial", "On 2024-08-01 I joined Northwind.",
                                  "Northwind"))
        result = memory.remember(**proposal("My employer is Aster.", kind=kind))
        new = result["claims"][0]
        assert new["status"] == "disputed" and new["update_kind"] == "assertion"
        assert not new["metadata"]["proposition_certificate"]["transition_entailment"]
        old = next(c for c in memory.memory.store.owner_claims("u")
                   if c.object_value == "Northwind")
        assert old.status == "disputed" and old.valid_end is None
        assert memory.memory.store.relations_for([new["id"]])[0].relation == "contradicts"


@pytest.mark.parametrize("kind,source", [
    ("correction", "I don't work at Aster."),
    ("correction", "I work at Aster."),
    ("retraction", "I work at Aster."),
    ("retraction", "If I don't work at Aster, I will move."),
])
def test_destructive_update_labels_cannot_override_source_meaning(tmp_path, kind, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        result = memory.remember(**proposal(source, kind=kind))
        assert result["claims"][0]["status"] == "quarantined"


def test_explicit_transition_survives_short_evidence_but_respects_model_veto(tmp_path):
    for veto in (False, True):
        with StandaloneMemory.offline(tmp_path / f"memory-{veto}.db") as memory:
            memory.remember(**fixture("u", "initial", "On 2024-08-01 I joined Northwind.",
                                      "Northwind"))
            data = proposal("On 2024-10-01 I left Northwind and joined Aster.", evidence="Aster",
                            hints={"transition_entailment": not veto})
            result = memory.remember(**data)
            claim = result["claims"][0]
            assert claim["status"] == ("disputed" if veto else "active")
            old = next(c for c in memory.memory.store.owner_claims("u")
                       if c.object_value == "Northwind")
            assert old.status == ("disputed" if veto else "superseded")


def test_full_source_is_restored_when_model_omits_source_content(tmp_path):
    data = proposal("In a hypothetical example, I work at Aster.", evidence="I work at Aster.")
    del data["cached_claims"]["claims"][0]["source_content"]
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = memory.remember(**data)["claims"][0]
        assert claim["source_content"] == data["messages"][0].content
        assert claim["status"] == "quarantined"


@pytest.mark.parametrize("proposed_field", ["valid_start", "valid_end"])
def test_model_only_date_does_not_control_current_state(tmp_path, proposed_field):
    data = proposal("I work at Aster.", kind="assertion")
    data["cached_claims"]["claims"][0][proposed_field] = "2030-01-01"
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = memory.remember(**data)["claims"][0]
        assert claim["status"] == "active"
        assert claim["valid_start"] is None and claim["valid_end"] is None
        temporal = claim["metadata"]["temporal_certificate"]
        assert temporal["detector"] == "temporal-grounding-v2"
        assert temporal["consistency"] == "insufficient"
        assert temporal["original_valid_start" if proposed_field == "valid_start"
                        else "original_valid_end"] == ensure_ms("2030-01-01")


@pytest.mark.parametrize("source", ["I worked at Aster.", "I used to work at Aster."])
def test_unbounded_past_only_state_is_not_current(tmp_path, source):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = memory.remember(**proposal(source, kind="assertion"))["claims"][0]
        assert claim["status"] == "quarantined"
        assert "past_only_state_without_supported_end" in claim["metadata"]["admission_reasons"]
        assert claim["valid_start"] is None and claim["valid_end"] is None


def test_past_only_state_with_explicit_interval_remains_historical(tmp_path):
    source = "From 2020-01-01 to 2021-01-01 I worked at Aster."
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = memory.remember(**proposal(source, kind="assertion"))["claims"][0]
        assert claim["status"] == "active"
        assert claim["valid_start"] == ensure_ms("2020-01-01")
        assert claim["valid_end"] == ensure_ms("2021-01-01")


@pytest.mark.parametrize("predicate,old,new,source", [
    ("employment.organization", "Aster", "Northwind",
     "Since 2024-10-01 I left Aster and joined Northwind."),
    ("residence.current", "Paris", "London",
     "Since 2024-10-01 I moved from Paris to London."),
])
def test_prior_transition_value_cannot_replace_current_state(
    tmp_path, predicate, old, new, source
):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        initial = fixture("u", "initial", f"Since 2024-08-01 I live or work at {old}.",
                          old, predicate=predicate, kind="assertion")
        memory.remember(**initial)
        wrong = fixture("u", "wrong", source, old, predicate=predicate,
                        date="2024-10-01", kind="transition")
        claim = memory.remember(**wrong)["claims"][0]
        assert claim["status"] == "quarantined"
        assert claim["metadata"]["proposition_certificate"]["transition_role"] == "prior"
        assert "object_is_prior_transition_value" in claim["metadata"]["admission_reasons"]
        assert memory.memory.store.stats()["counts"]["conflict_relations"] == 0
        accepted = fixture("u", "correct", source, new, predicate=predicate,
                           date="2024-10-01", kind="transition")
        current = memory.remember(**accepted)["claims"][0]
        assert current["status"] == "active"
        assert current["metadata"]["proposition_certificate"]["transition_role"] == "new"
        assert memory.memory.store.relations_for([current["id"]])[0].relation == "supersedes"


def test_change_cue_without_object_destination_does_not_authorize_state(tmp_path):
    with StandaloneMemory.offline(tmp_path / "memory.db") as memory:
        claim = memory.remember(**proposal(
            "I joined Northwind after Aster.", kind="transition",
        ))["claims"][0]
        assert claim["status"] == "quarantined"
        assert claim["metadata"]["proposition_certificate"]["transition_role"] == "unknown"
        assert "transition_destination_unverified" in claim["metadata"]["admission_reasons"]
