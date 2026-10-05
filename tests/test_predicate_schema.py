from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from stacmem import StandaloneMemory
from stacmem.config import AppConfig
from stacmem.contracts import ContractClaimExtractor, ContractPolicies, ContractQueryCompiler
from stacmem.models import Place, QueryFrame
from stacmem.predicate_schema import PredicateDefinition, PredicateSchemaConfig
from stacmem.prepared import policy_fingerprint
from stacmem.standalone_demo import fixture
from stacmem.time_utils import ensure_ms


def config(path, *, scope="global"):
    result = AppConfig()
    result.runtime.database_path = str(path)
    result.extraction.provider = "rule"
    result.embedding.provider = "hash"
    result.embedding.dimensions = 128
    result.rerank.provider = "none"
    result.predicate_schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name="contact.channel", aliases=("preferred.channel",),
        description="The person's preferred contact channel.", scope=scope,
    ),))
    return result


def payload(session="s1", value="email", *, place=None, date="2025-01-01"):
    location = f" in {place}" if place else ""
    return fixture(
        "u", session, f"Since {date}{location} I prefer {value}.", value,
        predicate="preferred.channel", kind="assertion", date=date, place=place,
    )


def search(memory, *, predicate="preferred.channel", date="2025-05-01", place=None):
    question = f"On {date}, what is my contact channel?"
    return memory.search(owner_id="u", query=question, frame=QueryFrame(
        raw_query=question, owner_id="u", target_subject="u", target_predicate=predicate,
        temporal_intent="as_of", query_time=ensure_ms(date),
        place=Place(name=place) if place else None,
    ))


@pytest.mark.parametrize("changes", [
    {"cardinality": "many"}, {"value_type": "number"}, {"scope": "condition"},
    {"functional": False}, {"name": "Bad Name"}, {"description": "  "},
    {"aliases": (" x ", "X")}, {"aliases": (" ",)},
])
def test_unsupported_schema_semantics_rejected(changes):
    with pytest.raises(ValidationError):
        PredicateDefinition(**{"name": "contact.channel", "description": "Contact", **changes})


@pytest.mark.parametrize("name,aliases", [
    ("employment.organization", ()), ("contact.channel", ("works_at",)),
    ("contact.channel", ("contact.channel",)),
])
def test_alias_collisions_rejected(name, aliases):
    schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name=name, aliases=aliases, description="Collision",
    ),))
    with pytest.raises(ValueError, match="collision"):
        ContractPolicies(schema)


def test_custom_cross_alias_collision_and_instance_isolation(tmp_path):
    custom = config(tmp_path / "db").predicate_schema
    policies = ContractPolicies(custom)
    assert policies.canonicalize(" Preferred.Channel ") == "contact.channel"
    assert "contact.channel" not in ContractPolicies().specs
    assert "contact.channel" not in ContractPolicies.specs
    with pytest.raises(TypeError):
        policies.aliases["works_at"] = "contact.channel"
    duplicate = PredicateDefinition(name="other.channel", description="Other",
                                    aliases=("contact.channel",))
    with pytest.raises(ValueError, match="collision"):
        ContractPolicies(PredicateSchemaConfig(predicates=(*custom.predicates, duplicate)))


def test_registered_alias_write_current_and_history(tmp_path):
    with StandaloneMemory(config(tmp_path / "db")) as memory:
        first = payload()
        first["cached_claims"]["claims"][0]["functional"] = False
        written = memory.remember(**first)
        assert written["claims"][0]["predicate"] == "contact.channel"
        assert written["claims"][0]["functional"] is True
        second = fixture("u", "s2", "On 2025-03-01 I switched from email to phone.", "phone",
                         predicate="contact.channel", date="2025-03-01")
        memory.remember(**second)
        assert [x["claim"]["object_value"] for x in search(memory)["claims"]] == ["phone"]
        old = search(memory, date="2025-02-01")
        assert [x["claim"]["object_value"] for x in old["claims"]] == ["email"]
        assert old["claims"][0]["claim"]["status"] == "superseded"
        question = "What is the history of my contact channel?"
        history = memory.search(owner_id="u", query=question, frame=QueryFrame(
            raw_query=question, owner_id="u", target_subject="u",
            target_predicate="contact.channel", temporal_intent="history",
        ))
        assert {x["claim"]["object_value"] for x in history["claims"]} == {"email", "phone"}


def test_custom_scope_isolates_places(tmp_path):
    with StandaloneMemory(config(tmp_path / "db", scope="place")) as memory:
        memory.remember(**payload(place="Bern"))
        memory.remember(**payload("s2", "phone", place="Rome"))
        assert [x["claim"]["object_value"] for x in search(memory, place="Bern")["claims"]] == [
            "email"
        ]
        assert [x["claim"]["object_value"] for x in search(memory, place="Rome")["claims"]] == [
            "phone"
        ]
        assert search(memory)["warnings"]


@pytest.mark.parametrize("target", ["unsupported.cost", None])
def test_unknown_query_returns_explicit_route_without_retrieval(tmp_path, monkeypatch, target):
    with StandaloneMemory(config(tmp_path / "db")) as memory:
        memory.remember(**payload())

        def forbidden(*args, **kwargs):
            raise AssertionError("Unsupported state query must not retrieve or call reader")

        monkeypatch.setattr(memory.memory, "search", forbidden)
        frame = QueryFrame(raw_query="cost", owner_id="u", target_predicate=target)
        result = memory.search(owner_id="u", query="cost", frame=frame)
        assert result["claims"] == []
        assert result["diagnostics"]["query_routing"] == "unsupported_predicate"
        assert memory.ask(owner_id="u", query="cost", frame=frame, backend=None)["abstain"]
        assert memory.source_search(owner_id="u", query="email")


def test_registration_does_not_bypass_source_or_factuality_checks(tmp_path):
    with StandaloneMemory(config(tmp_path / "db")) as memory:
        invalid = payload()
        invalid["cached_claims"]["claims"][0]["object_surface"] = "invented"
        assert memory.remember(**invalid)["claims"][0]["status"] == "quarantined"
        uncertain = fixture("u", "s2", "I might prefer phone.", "phone",
                            predicate="contact.channel", kind="assertion")
        grounding = uncertain["cached_claims"]["claims"][0]["proposition_grounding"]
        grounding["factuality"] = "nonfactual"
        assert memory.remember(**uncertain)["claims"][0]["status"] == "quarantined"
        assert search(memory)["claims"] == []


def test_unknown_claim_cannot_acquire_destructive_permissions(tmp_path):
    with StandaloneMemory.offline(tmp_path / "db") as memory:
        result = memory.remember(**payload())
        assert result["claims"][0]["status"] == "quarantined"
        assert result["claims"][0]["functional"] is False
        assert result["relations"] == []
        assert memory.source_search(owner_id="u", query="email")


def test_schema_bound_to_database_across_restarts(tmp_path):
    path = tmp_path / "db"
    with StandaloneMemory(config(path)) as memory:
        memory.remember(**payload())
        fingerprint = memory.status()["predicate_schema"]["fingerprint"]
    with StandaloneMemory(config(path)) as memory:
        assert memory.status()["predicate_schema"]["fingerprint"] == fingerprint
        assert search(memory)["claims"]
    with pytest.raises(ValueError, match="stored schema"):
        StandaloneMemory(config(path, scope="place"))
    with pytest.raises(ValueError, match="stored schema"):
        StandaloneMemory.offline(path)


def test_populated_database_without_schema_binding_requires_migration(tmp_path):
    path = tmp_path / "db"
    with StandaloneMemory.offline(path) as memory:
        memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM standalone_meta WHERE key='predicate_schema'")
    with pytest.raises(ValueError, match="Populated database lacks predicate_schema"):
        StandaloneMemory(config(path))
    with pytest.raises(ValueError, match="Populated database lacks predicate_schema"):
        StandaloneMemory.offline(path)


def test_prepared_recovery_uses_same_schema_and_no_provider(tmp_path, monkeypatch):
    path = tmp_path / "db"
    original_config = config(path)
    assert policy_fingerprint(original_config) != policy_fingerprint(config(path, scope="place"))
    with StandaloneMemory(original_config) as memory:
        def interrupt(**kwargs):
            raise RuntimeError("interrupt")
        monkeypatch.setattr(memory, "_commit_prepared", interrupt)
        with pytest.raises(RuntimeError, match="interrupt"):
            memory.remember(**payload())
    with StandaloneMemory(original_config) as memory:
        def no_embedding(*args, **kwargs):
            raise AssertionError("Must recover saved vectors")
        monkeypatch.setattr(memory.memory, "embed_drafts", no_embedding)
        recovered = memory.recover_prepared(owner_id="u", session_id="s1")
        assert recovered["claims"][0]["predicate"] == "contact.channel"
        assert search(memory)["claims"]


def test_model_prompts_and_compilation_use_same_registry(tmp_path):
    policies = ContractPolicies(config(tmp_path / "db").predicate_schema)

    class Backend:
        def __init__(self):
            self.last_usage = {}
            self.response = {}

        def call(self, system, user):
            assert policies.prompt() in system
            return self.response

    backend = Backend()
    request = payload()
    backend.response = request["cached_claims"]
    drafts = ContractClaimExtractor(backend, policies=policies).extract(
        owner_id="u", session_id="s1", messages=request["messages"],
    )
    assert drafts[0].object_value == "email"
    backend.response = {"target_subject": "self", "target_predicate": "preferred.channel",
                        "temporal_intent": "current"}
    frame = ContractQueryCompiler(backend, policies=policies).compile(
        owner_id="u", query="My current channel?", asked_at=ensure_ms("2025-01-01"),
    )
    assert frame.target_predicate == "contact.channel"


def test_config_roundtrip_and_unknown_schema_key(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('''[[predicate_schema.predicates]]
name = "contact.channel"
description = "Preferred contact channel"
aliases = ["preferred.channel"]
scope = "place"
''', encoding="utf-8")
    loaded = AppConfig.load(path)
    policy = ContractPolicies(loaded.predicate_schema).resolve("preferred.channel", False)
    assert policy.spatially_scoped
    assert json.loads(loaded.model_dump_json())["predicate_schema"]["predicates"][0]["name"] == (
        "contact.channel"
    )
    with pytest.raises(ValidationError):
        PredicateSchemaConfig.model_validate({"predicate": []})
