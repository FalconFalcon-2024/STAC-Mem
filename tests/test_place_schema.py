from __future__ import annotations

import json
import sqlite3

import pytest

from stacmem.config import AppConfig
from stacmem.contracts import scope_key, source_place
from stacmem.models import Place, QueryFrame, TemporalIntent
from stacmem.place_schema import PlaceDefinition, PlaceRegistry, PlaceSchemaConfig
from stacmem.standalone import StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.time_utils import ensure_ms


def schema(*definitions: PlaceDefinition) -> PlaceSchemaConfig:
    return PlaceSchemaConfig(places=definitions)


def test_trusted_aliases_share_one_scope_without_trusting_model_identity():
    registry = PlaceRegistry(schema(PlaceDefinition(
        place_id="ch/be/bern",
        name="Bern",
        aliases=("Berne", "伯尔尼"),
        hierarchy=("CH", "BE", "Bern"),
    )))
    places = [
        source_place(
            {"name": name, "place_id": "invented", "hierarchy": ["invented"]},
            f"I prefer trains in {name}.",
            registry=registry,
        )
        for name in ("Bern", "Berne", "伯尔尼")
    ]
    assert {scope_key(place) for place in places} == {"ch/be/bern"}
    assert [place.name for place in places] == ["Bern", "Berne", "伯尔尼"]
    assert all(place.hierarchy == ["CH", "BE", "Bern"] for place in places)


def test_unknown_grounded_place_remains_literal():
    place = source_place(
        {"name": "Basel", "place_id": "model-invented"},
        "I prefer trains in Basel.",
        registry=PlaceRegistry(),
    )
    assert place.place_id is None
    assert scope_key(place) == "basel"


def test_registry_rejects_alias_and_identity_collisions():
    with pytest.raises(ValueError, match="alias collision"):
        PlaceRegistry(schema(
            PlaceDefinition(place_id="a", name="Bern"),
            PlaceDefinition(place_id="b", name="BERN"),
        ))
    with pytest.raises(ValueError, match="Duplicate place_id"):
        PlaceRegistry(schema(
            PlaceDefinition(place_id="a", name="Bern"),
            PlaceDefinition(place_id="A", name="Basel"),
        ))


def test_alias_order_does_not_change_manifest_or_database_binding(tmp_path):
    left = PlaceRegistry(schema(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("Berne", "伯尔尼")
    )))
    right = PlaceRegistry(schema(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("伯尔尼", "Berne")
    )))
    assert left.manifest() == right.manifest()
    config = AppConfig()
    config.runtime.database_path = str(tmp_path / "memory.sqlite3")
    config.embedding.provider = "hash"
    config.embedding.dimensions = 128
    config.extraction.provider = "rule"
    config.rerank.provider = "none"
    config.place_schema = schema(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("Berne", "伯尔尼")
    ))
    with StandaloneMemory(config):
        pass
    with sqlite3.connect(config.runtime.database_path) as database:
        for table in ("standalone_meta", "ledger_meta"):
            raw = database.execute(
                f"SELECT value FROM {table} WHERE key='place_schema'"
            ).fetchone()[0]
            manifest = json.loads(raw)
            manifest["places"][0]["aliases"].reverse()
            database.execute(
                f"UPDATE {table} SET value=? WHERE key='place_schema'",
                (json.dumps(manifest, ensure_ascii=False),),
            )
    with StandaloneMemory(config):
        pass
    config.place_schema = schema(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("伯尔尼", "Berne")
    ))
    with StandaloneMemory(config):
        pass


def test_place_schema_is_bound_to_database(tmp_path):
    path = tmp_path / "memory.sqlite3"
    config = AppConfig()
    config.runtime.database_path = str(path)
    config.embedding.provider = "hash"
    config.embedding.dimensions = 128
    config.extraction.provider = "rule"
    config.rerank.provider = "none"
    config.place_schema = schema(PlaceDefinition(place_id="ch/be/bern", name="Bern"))
    with StandaloneMemory(config) as memory:
        status = memory.status()["place_schema"]
        assert status["manifest"]["places"][0]["place_id"] == "ch/be/bern"

    changed = config.model_copy(deep=True)
    changed.place_schema = schema(PlaceDefinition(place_id="ch/bs/basel", name="Basel"))
    with pytest.raises(ValueError, match="place_schema differs"):
        StandaloneMemory(changed)


def test_aliases_share_conflict_slot_and_structured_query_scope(tmp_path):
    config = AppConfig()
    config.runtime.database_path = str(tmp_path / "memory.sqlite3")
    config.embedding.provider = "hash"
    config.embedding.dimensions = 128
    config.extraction.provider = "rule"
    config.rerank.provider = "none"
    config.place_schema = schema(PlaceDefinition(
        place_id="ch/be/bern", name="Bern", aliases=("Berne", "伯尔尼")
    ))
    with StandaloneMemory(config) as memory:
        memory.remember(**fixture(
            "u", "s1", "In Berne I prefer bus.", "bus",
            predicate="preference.commute_mode", place="Berne", kind="assertion",
        ))
        memory.remember(**fixture(
            "u", "s2", "In Bern I changed to tram.", "tram",
            predicate="preference.commute_mode", place="Bern", kind="transition",
            date="2024-10-15",
        ))
        query = "How do I commute in 伯尔尼?"
        result = memory.search(owner_id="u", query=query, frame=QueryFrame(
            raw_query=query,
            owner_id="u",
            target_subject="u",
            target_predicate="preference.commute_mode",
            temporal_intent=TemporalIntent.CURRENT,
            query_time=ensure_ms("2025-01-01"),
            place=Place(name="伯尔尼", role="scope"),
        ))
        assert [item["claim"]["object_value"] for item in result["claims"]] == ["tram"]
        assert result["claims"][0]["claim"]["place"]["place_id"] == "ch/be/bern"
