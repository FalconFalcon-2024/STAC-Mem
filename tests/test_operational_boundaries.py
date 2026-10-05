from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from stacmem import StandaloneMemory
from stacmem.config import AppConfig
from stacmem.contracts import ContractPolicies
from stacmem.pipeline import StacMemory, embedding_space_manifest
from stacmem.place_schema import PlaceDefinition, PlaceSchemaConfig
from stacmem.predicate_schema import PredicateDefinition, PredicateSchemaConfig
from stacmem.prepared import policy_fingerprint
from stacmem.standalone_demo import fixture


def offline_config(path, *, dimensions=128, space_id=None, retention="until_commit"):
    config = AppConfig()
    config.runtime.database_path = str(path)
    config.extraction.provider = "rule"
    config.embedding.provider = "hash"
    config.embedding.dimensions = dimensions
    config.embedding.space_id = space_id
    config.rerank.provider = "none"
    config.recovery.prepared_retention = retention
    return config


def test_database_rejects_mixed_dimensions_and_preserves_original_vectors(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory(offline_config(path)) as memory:
        memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
    with pytest.raises(ValueError, match="Embedding space differs"):
        StandaloneMemory(offline_config(path, dimensions=64))
    with sqlite3.connect(path) as database:
        assert database.execute(
            "SELECT DISTINCT vector_dimensions FROM claims"
        ).fetchall() == [(128,)]


def test_same_dimensions_but_different_space_id_is_rejected(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory(offline_config(path, space_id="space-a")) as memory:
        memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
    with pytest.raises(ValueError, match="Embedding space differs"):
        StandaloneMemory(offline_config(path, space_id="space-b"))


def test_empty_database_can_be_reconfigured_before_first_claim(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory(offline_config(path, dimensions=64)):
        pass
    with StandaloneMemory(offline_config(path, dimensions=128)) as memory:
        assert memory.status()["ledger"]["counts"]["claims"] == 0
        assert memory.memory.store.embedding_space()["dimensions"] == 128


def test_endpoint_is_part_of_openai_compatible_space_identity(tmp_path):
    left = offline_config(tmp_path / "left.db")
    left.embedding.provider = "openai_compatible"
    left.embedding.model = "same-model"
    left.embedding.base_url = "https://left.invalid/v1"
    right = left.model_copy(deep=True)
    right.embedding.base_url = "https://right.invalid/v1"
    assert embedding_space_manifest(left) != embedding_space_manifest(right)


def test_legacy_claims_without_binding_require_explicit_migration(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory(offline_config(path)) as memory:
        memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
    with sqlite3.connect(path) as database:
        database.execute("DELETE FROM ledger_meta WHERE key='embedding_space'")
    with pytest.raises(ValueError, match="claims or prepared batches"):
        StandaloneMemory(offline_config(path))


def test_legacy_runtime_profile_is_rejected_without_half_migration(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory(offline_config(path)) as memory:
        memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
    with sqlite3.connect(path) as database:
        database.execute(
            "UPDATE standalone_meta SET value=? WHERE key='profile'",
            ("standalone-source-v1+contracts-v15",),
        )
    with pytest.raises(ValueError, match="Legacy database detected"):
        StandaloneMemory(offline_config(path))
    with sqlite3.connect(path) as database:
        assert database.execute(
            "SELECT value FROM standalone_meta WHERE key='profile'"
        ).fetchone()[0] == "standalone-source-v1+contracts-v15"
        assert database.execute("SELECT count(*) FROM claims").fetchone()[0] == 1


def test_prepared_batch_blocks_embedding_space_rebinding(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    first = offline_config(path, space_id="space-a")
    with StandaloneMemory(first) as memory:
        def stop_before_commit(**kwargs):
            raise RuntimeError("simulated stop after durable preparation")

        monkeypatch.setattr(memory, "_commit_prepared", stop_before_commit)
        with pytest.raises(RuntimeError, match="durable preparation"):
            memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
        assert memory.status()["recovery"]["prepared_batches"] == 1

    with pytest.raises(ValueError, match="Embedding space differs"):
        StandaloneMemory(offline_config(path, space_id="space-b"))
    with sqlite3.connect(path) as database:
        stored = database.execute(
            "SELECT value FROM ledger_meta WHERE key='embedding_space'"
        ).fetchone()[0]
        assert '"space_id":"space-a"' in stored


def test_prepared_payload_carries_embedding_space_fingerprint(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    first = offline_config(path, space_id="space-a")
    with StandaloneMemory(first) as memory:
        def stop_after_prepare(**kwargs):
            raise RuntimeError("stop after prepare")

        monkeypatch.setattr(memory, "_commit_prepared", stop_after_prepare)
        with pytest.raises(RuntimeError, match="stop after prepare"):
            memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))

    with sqlite3.connect(path) as database:
        row = database.execute(
            "SELECT value FROM ledger_meta WHERE key='embedding_space'"
        ).fetchone()[0]
        changed = json.loads(row)
        changed["space_id"] = "space-b"
        database.execute(
            "UPDATE ledger_meta SET value=? WHERE key='embedding_space'",
            (json.dumps(changed, sort_keys=True, separators=(",", ":")),),
        )

    with (
        StandaloneMemory.recovery(offline_config(path, space_id="space-b")) as recovery,
        pytest.raises(ValueError, match="embedding_space_fingerprint mismatch"),
    ):
        recovery.recover_prepared(owner_id="u", session_id="s1")


def test_default_prepared_journal_is_removed_atomically_after_commit(tmp_path):
    path = tmp_path / "state.db"
    request = fixture("u", "s1", "I joined Aster.", "Aster")
    with StandaloneMemory(offline_config(path)) as memory:
        first = memory.remember(**request)
        assert first["status"] == "committed"
        assert memory.status()["recovery"]["prepared_batches"] == 0
        assert memory.remember(**request)["replayed"] is True
    with sqlite3.connect(path) as database:
        assert database.execute("SELECT count(*) FROM prepared_batches").fetchone()[0] == 0


def test_debug_retention_can_keep_committed_prepared_batch(tmp_path):
    path = tmp_path / "state.db"
    with StandaloneMemory(offline_config(path, retention="forever")) as memory:
        memory.remember(**fixture("u", "s1", "I joined Aster.", "Aster"))
        recovery = memory.status()["recovery"]
        assert recovery["prepared_retention"] == "forever"
        assert recovery["prepared_batches"] == 1


@pytest.mark.parametrize("section", [
    {"runtime": {"request_timeout_second": 3}},
    {"extraction": {"base_urll": "https://invalid.example/v1"}},
    {"embedding": {"send_dimension": False}},
    {"retrieval": {"candidate_pole": 10}},
    {"conflict": {"infer_transitions": False}},
    {"recovery": {"prepared_retention": "sometimes"}},
])
def test_config_typos_and_unknown_values_are_rejected(section):
    with pytest.raises(ValidationError):
        AppConfig.model_validate(section)


def test_assignment_and_environment_override_are_revalidated(tmp_path, monkeypatch):
    config = AppConfig()
    with pytest.raises(ValidationError):
        config.embedding.dimensions = -1
    with pytest.raises(ValidationError):
        config.embedding.space_id = "   "
    settings = tmp_path / "config.toml"
    settings.write_text("[embedding]\nprovider='hash'\n", encoding="utf-8")
    monkeypatch.setenv("STACMEM_EMBEDDING_DIMENSIONS", "-1")
    with pytest.raises(ValidationError):
        AppConfig.load(settings)


def test_public_factory_consumes_custom_predicate_schema(tmp_path):
    config = offline_config(tmp_path / "state.db")
    config.predicate_schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name="contact.channel", description="Preferred channel.",
        aliases=("preferred.channel",), scope="place",
    ),))
    with StacMemory.from_app_config(config) as memory:
        policies = memory.conflict.policies
        assert policies.canonicalize("preferred.channel") == "contact.channel"
        assert policies.resolve("contact.channel", False).functional
        assert policies.resolve("contact.channel", False).spatially_scoped


def test_direct_engine_binds_predicate_scope_and_place_aliases(tmp_path):
    path = tmp_path / "state.db"
    config = offline_config(path)
    config.predicate_schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name="contact.channel", description="Preferred channel.", scope="global",
    ),))
    config.place_schema = PlaceSchemaConfig(places=(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("Berne",),
    ),))
    with StacMemory.from_app_config(config) as memory:
        assert memory.store._conn.execute(
            "SELECT count(*) FROM ledger_meta WHERE key IN ('predicate_schema','place_schema')"
        ).fetchone()[0] == 2

    changed = config.model_copy(deep=True)
    changed.predicate_schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name="contact.channel", description="Preferred channel.", scope="place",
    ),))
    with pytest.raises(ValueError, match="predicate_schema differs"):
        StacMemory.from_app_config(changed)
    changed = config.model_copy(deep=True)
    changed.place_schema = PlaceSchemaConfig(places=(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("伯尔尼",),
    ),))
    with pytest.raises(ValueError, match="place_schema differs"):
        StacMemory.from_app_config(changed)


def test_direct_engine_rejects_unbound_populated_legacy_ledger(tmp_path):
    path = tmp_path / "state.db"
    config = offline_config(path)
    with StandaloneMemory(config) as memory:
        memory.remember(**fixture("u", "s", "I joined Aster.", "Aster"))
    with sqlite3.connect(path) as database:
        database.execute("DELETE FROM ledger_meta WHERE key IN ('predicate_schema','place_schema')")
        database.execute(
            "DELETE FROM standalone_meta WHERE key IN ('predicate_schema','place_schema')"
        )
    with pytest.raises(ValueError, match="Populated database has no predicate_schema"):
        StacMemory.from_app_config(config)


def test_existing_standalone_binding_bootstraps_shared_ledger_binding(tmp_path):
    path = tmp_path / "state.db"
    config = offline_config(path)
    with StandaloneMemory(config) as memory:
        memory.remember(**fixture("u", "s", "I joined Aster.", "Aster"))
    with sqlite3.connect(path) as database:
        database.execute("DELETE FROM ledger_meta WHERE key IN ('predicate_schema','place_schema')")
    with StacMemory.from_app_config(config) as memory:
        assert memory.store.stats()["counts"]["claims"] == 1
        assert memory.store._conn.execute(
            "SELECT count(*) FROM ledger_meta WHERE key IN ('predicate_schema','place_schema')"
        ).fetchone()[0] == 2


def test_empty_direct_engine_bootstraps_to_source_runtime(tmp_path):
    path = tmp_path / "state.db"
    config = offline_config(path)
    config.predicate_schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name="contact.channel", description="Preferred channel.", scope="global",
    ),))
    with StacMemory.from_app_config(config):
        pass
    with StandaloneMemory(config) as memory:
        result = memory.remember(**fixture("u", "s", "I prefer email.", "email",
                                            predicate="contact.channel", kind="assertion"))
        assert result["status"] == "committed"
        assert result["claims"][0]["status"] == "active"
    with StacMemory.from_app_config(config) as memory:
        assert memory.store.stats()["counts"]["claims"] == 1


def test_empty_engine_schema_mismatch_fails_before_creating_source_tables(tmp_path):
    path = tmp_path / "state.db"
    original = offline_config(path)
    original.place_schema = PlaceSchemaConfig(places=(PlaceDefinition(
        place_id="ch/be/bern", name="Bern",
    ),))
    with StacMemory.from_app_config(original):
        pass
    changed = offline_config(path)
    with pytest.raises(ValueError, match="place_schema differs"):
        StandaloneMemory(changed)
    with sqlite3.connect(path) as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='standalone_meta'"
        ).fetchone()


def test_populated_direct_ledger_requires_source_migration(tmp_path):
    path = tmp_path / "state.db"
    config = offline_config(path)
    from stacmem.contracts import compile_claim_payload

    with StacMemory.from_app_config(config) as memory:
        data = fixture("u", "s", "I joined Aster.", "Aster")
        memory.ingest_drafts(compile_claim_payload(
            data["cached_claims"], owner_id="u", session_id="s", messages=data["messages"]
        ))
    with pytest.raises(ValueError, match="only an empty direct engine"):
        StandaloneMemory(config)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='standalone_meta'"
        ).fetchone()


def test_shared_schema_binding_does_not_write_half_a_pair(tmp_path):
    path = tmp_path / "state.db"
    config = offline_config(path)
    config.place_schema = PlaceSchemaConfig(places=(PlaceDefinition(
        place_id="ch/be/bern", name="Bern",
    ),))
    with StacMemory.from_app_config(config):
        pass
    with sqlite3.connect(path) as database:
        database.execute("DELETE FROM ledger_meta WHERE key='predicate_schema'")
    changed = config.model_copy(deep=True)
    changed.place_schema = PlaceSchemaConfig(places=(PlaceDefinition(
        place_id="ch/bs/basel", name="Basel",
    ),))
    with pytest.raises(ValueError, match="place_schema differs"):
        StacMemory.from_app_config(changed)
    with sqlite3.connect(path) as database:
        assert database.execute(
            "SELECT count(*) FROM ledger_meta WHERE key='predicate_schema'"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "name", ["language.spoken", "plan.travel", "item.location", "appointment.time"]
)
def test_inactive_legacy_names_can_be_registered(name):
    schema = PredicateSchemaConfig(predicates=(PredicateDefinition(
        name=name, description="Application-owned state.",
    ),))
    assert name in ContractPolicies(schema).specs


def test_policy_fingerprint_tracks_semantics_not_operational_settings(tmp_path):
    original = offline_config(tmp_path / "state.db")
    operational = original.model_copy(deep=True)
    operational.runtime.request_timeout_seconds = 1
    operational.recovery.prepared_retention = "forever"
    assert policy_fingerprint(original) == policy_fingerprint(operational)
    changed = original.model_copy(deep=True)
    changed.conflict.infer_transition_end = False
    assert policy_fingerprint(original) != policy_fingerprint(changed)


def test_public_runtime_rejects_silently_ignored_correctness_switches(tmp_path):
    config = offline_config(tmp_path / "state.db")
    config.admission.enabled = False
    with pytest.raises(ValueError, match="fixed correctness semantics"):
        StacMemory.from_app_config(config)
