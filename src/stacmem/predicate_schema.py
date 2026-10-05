"""Validated application-owned extensions to the state predicate vocabulary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredicateDefinition(BaseModel):
    """Only semantics implemented by the single-value text state engine are accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
    description: str = Field(min_length=1, max_length=500)
    aliases: tuple[str, ...] = ()
    cardinality: Literal["one"] = "one"
    value_type: Literal["text"] = "text"
    scope: Literal["global", "place"] = "global"

    @field_validator("description")
    @classmethod
    def nonblank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Predicate description must not be blank")
        return value.strip()

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.casefold().strip().replace(" ", "_") for value in values)
        if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("Aliases must be nonempty and unique after normalization")
        return tuple(sorted(normalized))


class PredicateSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicates: tuple[PredicateDefinition, ...] = ()
