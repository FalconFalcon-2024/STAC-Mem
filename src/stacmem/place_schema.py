"""Application-owned place identities used to normalize grounded place mentions."""

from __future__ import annotations

from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Place


def normalize_place_name(value: str) -> str:
    return " ".join(value.casefold().strip().split())


class PlaceDefinition(BaseModel):
    """A trusted identity and its source-language names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    place_id: str = Field(min_length=1, max_length=300, pattern=r"^\S+$")
    name: str = Field(min_length=1, max_length=300)
    aliases: tuple[str, ...] = ()
    hierarchy: tuple[str, ...] = ()

    @field_validator("place_id", "name")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Place identity fields must not be blank")
        return value

    @field_validator("aliases", "hierarchy")
    @classmethod
    def strip_sequence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("Place aliases and hierarchy members must not be blank")
        return cleaned

    @field_validator("aliases")
    @classmethod
    def sort_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(values, key=normalize_place_name))

    @model_validator(mode="after")
    def unique_names(self) -> PlaceDefinition:
        names = [normalize_place_name(value) for value in (self.name, *self.aliases)]
        if len(names) != len(set(names)):
            raise ValueError("Place name and aliases must be unique after normalization")
        return self


class PlaceSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    places: tuple[PlaceDefinition, ...] = ()


class PlaceRegistry:
    """Resolve only literal source mentions against application-trusted aliases."""

    def __init__(self, config: PlaceSchemaConfig | None = None) -> None:
        config = config or PlaceSchemaConfig()
        by_name: dict[str, PlaceDefinition] = {}
        ids: set[str] = set()
        for definition in sorted(config.places, key=lambda item: item.place_id):
            place_id = definition.place_id.casefold()
            if place_id in ids:
                raise ValueError(f"Duplicate place_id: {definition.place_id}")
            ids.add(place_id)
            for value in (definition.name, *definition.aliases):
                key = normalize_place_name(value)
                if key in by_name:
                    raise ValueError(f"Place alias collision: {value}")
                by_name[key] = definition
        self._by_name = MappingProxyType(by_name)
        self._definitions = tuple(sorted(config.places, key=lambda item: item.place_id))

    def resolve(self, place: Place | None) -> Place | None:
        if place is None or not place.name:
            return place
        definition = self._by_name.get(normalize_place_name(place.name))
        if definition is None:
            return place
        return Place(
            place_id=definition.place_id,
            name=place.name,
            hierarchy=list(definition.hierarchy),
            role=place.role,
        )

    def manifest(self) -> dict:
        return {
            "version": "trusted-place-registry-v1",
            "places": [item.model_dump(mode="json") for item in self._definitions],
        }


def canonical_place_manifest(value: dict) -> dict:
    """Treat alias order as presentation, including in older stored manifests."""
    return {
        "version": value["version"],
        "places": [
            {
                **item,
                "aliases": sorted(item["aliases"], key=normalize_place_name),
            }
            for item in sorted(value["places"], key=lambda place: place["place_id"])
        ],
    }
