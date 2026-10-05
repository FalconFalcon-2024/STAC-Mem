from __future__ import annotations

import pytest

from stacmem.contracts import (
    ContractError,
    ContractPolicies,
    compile_claim_payload,
    compile_query_payload,
    scope_key,
    source_place,
)
from stacmem.models import ClaimStatus, Message, Place
from stacmem.state_runtime import install_contracts as install_contract_runtime
from stacmem.time_utils import ensure_ms


def install_contracts(memory):
    """Switch the low-level test fixture to the fixed public state semantics."""
    memory.config.admission.enabled = True
    memory.config.admission.mode = "proposition_v2"
    memory.config.temporal_grounding.enabled = True
    memory.config.temporal_grounding.mode = "certificate_v2"
    memory.config.conflict.transition_mode = "proposition_v2"
    memory.config.conflict.temporal_relation_mode = "valid_interval_v2"
    return install_contract_runtime(memory)


def extract(
    source,
    value,
    *,
    surface=None,
    place=None,
    predicate="residence.current",
    start="2025-01-01",
    kind="assertion",
    message_id="s:0",
    stamp="2025-04-01",
):
    raw = {
        "subject": "u",
        "predicate": predicate,
        "object_value": value,
        "valid_start": start,
        "valid_end": None,
        "update_kind": kind,
        "functional": True,
        "source_content": source,
        "source_message_ids": [message_id],
        "place": {"name": place, "role": "scope"} if place else None,
        "proposition_grounding": {
            "evidence_span": source,
            "subject_alignment": "aligned",
            "factuality": "asserted",
            "transition_entailment": kind == "transition",
        },
    }
    if surface is not None:
        raw["object_surface"] = surface
    messages = [
        Message(
            sender_id="u",
            role="user",
            timestamp=ensure_ms(stamp),
            content=source,
            message_id=message_id,
        )
    ]
    return compile_claim_payload(
        {"claims": [raw]}, owner_id="u", session_id=message_id.split(":")[0], messages=messages
    )[0]


def query(text, *, place=None):
    return compile_query_payload(
        {
            "temporal_intent": "as_of",
            "target_subject": "self",
            "target_predicate": "commute.preferred_weekday" if place else "residence.city",
            "place": {"name": place, "role": "scope"} if place else None,
        },
        owner_id="u",
        query=text,
        asked_at=ensure_ms("2026-01-01"),
    )


def search(memory, text, place=None):
    return memory.search(owner_id="u", query=text, frame=query(text, place=place))


@pytest.mark.parametrize(
    "source,surface,normalized",
    [
        ("我住在合肥。", "合肥", "Hefei"),
        ("我选择自行车。", "自行车", "bicycle"),
        ("I live in Bern.", "Bern", "BERN_CANONICAL_ID"),
    ],
)
def test_source_value_is_operational_and_normalization_is_unverified(source, surface, normalized):
    draft = extract(source, normalized, surface=surface)
    cert = draft.metadata["source_contract"]
    assert draft.object_value == surface
    assert cert["proposed_normalized_value"] == normalized
    assert cert["canonical_identity_verified"] is False
    assert cert["status"] == "validated"
    start, end = cert["message_codepoint_span"]
    assert source[start:end] == surface


def test_scope_key_prefers_trusted_place_identity_over_display_name():
    assert scope_key(Place(place_id="CH/BE/Bern", name="Bern")) == "ch/be/bern"
    assert scope_key(Place(hierarchy=["CH", "BE", "Bern"], name="Berne")) == "ch/be/bern"
    assert scope_key(Place(name=" BERN ")) == "bern"


def test_translated_legacy_value_without_surface_stays_quarantined(memory):
    install_contracts(memory)
    result = memory.ingest_drafts([extract("我住在合肥。", "Hefei")])[0]
    assert result.claim.status == ClaimStatus.QUARANTINED
    assert "object_surface_not_in_source" in result.admission_reasons


def test_source_message_forgery_and_substring_collision_rejected():
    draft = extract("Joanna lives here.", "Ann")
    assert draft.metadata["source_contract"]["status"] == "rejected"
    payload = {
        "claims": [
            {
                "subject": "u",
                "predicate": "residence.current",
                "object_value": "Rome",
                "source_content": "I live in Rome.",
                "source_message_ids": ["invented"],
            }
        ]
    }
    messages = [Message(sender_id="u", role="user", timestamp=1, content="I live in Bern.")]
    result = compile_claim_payload(payload, owner_id="u", session_id="s", messages=messages)[0]
    assert "invalid_source_message_ids" in result.metadata["source_contract"]["reasons"]


@pytest.mark.parametrize("role,sender", [("assistant", "agent"), ("tool", "tool"),
                                          ("user", "another-user")])
def test_non_user_source_cannot_authorize_user_state(memory, role, sender):
    install_contracts(memory)
    source = "The user works at Aster."
    payload = {
        "claims": [{
            "subject": "u", "predicate": "employment.organization",
            "object_value": "Aster", "object_surface": "Aster",
            "source_content": source, "source_message_ids": ["s:0"],
            "proposition_grounding": {"evidence_span": source, "subject_alignment": "aligned",
                                      "factuality": "asserted"},
        }]
    }
    messages = [Message(sender_id=sender, role=role, timestamp=1, content=source,
                        message_id="s:0")]
    draft = compile_claim_payload(payload, owner_id="u", session_id="s", messages=messages)[0]
    outcome = memory.ingest_drafts([draft])[0]
    assert outcome.claim.status == ClaimStatus.QUARANTINED
    assert "non_authoritative_source_role" in outcome.admission_reasons


def test_mixed_user_and_agent_references_cannot_authorize_state(memory):
    install_contracts(memory)
    source = "I work at Aster."
    payload = {"claims": [{
        "subject": "u", "predicate": "employment.organization",
        "object_value": "Aster", "object_surface": "Aster", "source_content": source,
        "source_message_ids": ["s:0", "s:1"],
        "proposition_grounding": {"evidence_span": source, "subject_alignment": "aligned",
                                  "factuality": "asserted"},
    }]}
    messages = [
        Message(sender_id="u", role="user", timestamp=1, content=source, message_id="s:0"),
        Message(sender_id="agent", role="assistant", timestamp=2, content=source,
                message_id="s:1"),
    ]
    draft = compile_claim_payload(payload, owner_id="u", session_id="s", messages=messages)[0]
    outcome = memory.ingest_drafts([draft])[0]
    assert outcome.claim.status == ClaimStatus.QUARANTINED
    assert "non_authoritative_source_role" in outcome.admission_reasons


def test_source_contract_does_not_bypass_factuality_gate(memory):
    install_contracts(memory)
    draft = extract("If I moved to Bern, it would be hypothetical.", "Bern", kind="transition")
    draft.metadata["proposition_grounding"]["factuality"] = "nonfactual"
    assert memory.ingest_drafts([draft])[0].claim.status == ClaimStatus.QUARANTINED


def test_model_geometry_not_treated_as_verified_location():
    result = source_place(
        {
            "name": "Bern",
            "aliases": ["Paris"],
            "latitude": 999,
            "hierarchy": ["country", "city"],
            "role": "scope",
        },
        "in Bern",
    )
    assert result.name == "Bern" and result.latitude is None and result.hierarchy == []
    assert result.aliases == []
    assert source_place({"name": "Paris"}, "in Bern") is None


def test_query_date_repaired_and_unrequested_knowledge_cutoff_removed():
    payload = {
        "temporal_intent": "as_of",
        "query_time": "2024-08-07T15:57:27.564Z",
        "knowledge_time": "2025-06-15",
        "target_predicate": "residence.location",
    }
    result = compile_query_payload(
        payload, owner_id="u", query="Where did I live on 2025-06-15?", asked_at=1
    )
    assert result.query_time == ensure_ms("2025-06-15")
    assert result.knowledge_time is None
    assert result.target_predicate == "residence.current"
    assert result.metadata["query_contract"]["original"]["query_time"] == payload["query_time"]


@pytest.mark.parametrize(
    "text",
    [
        "As known on 2025-08-01, where did I live on 2025-06-15?",
        "截至2025-08-01知道的信息，我在2025-06-15住在哪里？",
    ],
)
def test_explicit_knowledge_and_valid_time_remain_separate(text):
    result = query(text)
    assert result.knowledge_time == ensure_ms("2025-08-01")
    assert result.query_time == ensure_ms("2025-06-15")


@pytest.mark.parametrize(
    "text",
    [
        "Where was I on 2025-01-01 or 2025-02-01?",
        "Where was I yesterday?",
        "Where was I on 2025-01-01T13:00:00?",
    ],
)
def test_unsupported_temporal_queries_fail_explicitly(text):
    with pytest.raises(ContractError):
        query(text)


def test_place_scope_is_schema_defined_not_predicate_substring():
    policies = ContractPolicies()
    assert policies.resolve("commute.preferred_mode", True).spatially_scoped
    assert not policies.resolve("residence.current", True).spatially_scoped
    assert not policies.resolve("mystery.preference", True).functional


def test_different_places_same_value_retraction_does_not_cross_scope(memory):
    install_contracts(memory)
    for i, city in enumerate(["Bern", "Basel"]):
        memory.ingest_drafts(
            [
                extract(
                    f"Since 2025-01-01, in {city} I prefer bus.",
                    "bus",
                    place=city,
                    predicate="commute.preferred_mode",
                    message_id=f"s{i}:0",
                )
            ]
        )
    retract = extract(
        "I no longer prefer bus in Bern.",
        "bus",
        place="Bern",
        kind="retraction",
        predicate="commute.preferred_mode",
        stamp="2025-05-01",
    )
    outcome = memory.ingest_drafts([retract])[0]
    baseline = [c for c in memory.store.owner_claims("u") if c.place.name == "Basel"]
    assert baseline[0].status == ClaimStatus.ACTIVE
    assert any(r.relation.value == "coexists" for r in outcome.relations)


def test_local_preference_update_preserves_other_place_and_has_no_false_conflict(memory):
    install_contracts(memory)
    drafts = [
        extract(
            "Since 2025-01-01, in Bern I prefer bus.",
            "bus",
            place="Bern",
            predicate="commute.preferred_mode",
            message_id="s0:0",
        ),
        extract(
            "Since 2025-01-01, in Basel I prefer metro.",
            "metro",
            place="Basel",
            predicate="commute.preferred_mode",
            message_id="s1:0",
        ),
        extract(
            "Starting on 2025-04-01, in Bern I changed to taxi.",
            "taxi",
            place="Bern",
            predicate="commute.preferred_mode",
            start="2025-04-01",
            kind="transition",
            message_id="s2:0",
        ),
    ]
    memory.ingest_drafts(drafts)
    for city, expected in [("Bern", "taxi"), ("Basel", "metro")]:
        result = search(memory, f"On 2025-05-15 in {city}, how do I commute?", city)
        assert [c.claim.object_value for c in result.claims] == [expected]
        assert result.diagnostics["unresolved_slots"] == 0
    old = search(memory, "On 2025-02-01 in Bern, how do I commute?", "Bern")
    assert [c.claim.object_value for c in old.claims] == ["bus"]


@pytest.mark.parametrize(
    "source", ["截至2025-05-01，我仍然住在合肥。", "As of 2025-05-01, I still live in Bern."]
)
def test_still_observation_supports_day_but_not_unbounded_history_or_future(memory, source):
    install_contracts(memory)
    value = "合肥" if "合肥" in source else "Bern"
    memory.ingest_drafts([extract(source, value, start="2025-05-01")])
    for day, expected in [("2025-05-01", [value]), ("2024-03-15", []), ("2025-05-02", [])]:
        result = search(memory, f"Where did I live on {day}?")
        assert [c.claim.object_value for c in result.claims] == expected


def test_observation_does_not_end_prior_supported_state(memory):
    install_contracts(memory)
    memory.ingest_drafts([extract("Since 2025-01-01, I live in Bern.", "Bern", stamp="2025-01-01")])
    memory.ingest_drafts(
        [extract("As of 2025-05-01, I still live in Bern.", "Bern", stamp="2025-05-01")]
    )
    for date in ["2025-02-01", "2025-05-01", "2025-06-01"]:
        assert [
            c.claim.object_value for c in search(memory, f"Where did I live on {date}?").claims
        ] == ["Bern"]


def test_true_before_statement_is_not_changed_to_observation(memory):
    install_contracts(memory)
    outcome = memory.ingest_drafts(
        [extract("Before 2025-05-01, I lived in Bern.", "Bern", start=None)]
    )[0]
    assert outcome.claim.valid_end == ensure_ms("2025-05-01")
    assert "temporal_observation" not in outcome.claim.metadata
    assert search(memory, "Where did I live on 2025-05-01?").claims == []


def test_relative_date_cannot_silently_become_current():
    with pytest.raises(ContractError, match="Relative dates"):
        compile_query_payload(
            {"temporal_intent": "current"},
            owner_id="u",
            query="Where did I live yesterday?",
            asked_at=ensure_ms("2026-01-01"),
        )


def test_unknown_scope_does_not_become_global_preference(memory):
    install_contracts(memory)
    memory.ingest_drafts(
        [extract("I prefer bus.", "bus", predicate="commute.preferred_mode", place=None)]
    )
    frame = query("On 2025-05-01, how did I commute?")
    frame.target_predicate = "preference.commute_mode"
    result = memory.search(owner_id="u", query=frame.raw_query, frame=frame)
    assert result.claims == []
    assert result.diagnostics["scope_unverified"] == 1
    assert any("unverified place" in warning for warning in result.warnings)
