"""Typed configuration loaded from secret-free TOML files."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .place_schema import PlaceSchemaConfig
from .predicate_schema import PredicateSchemaConfig


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeConfig(StrictConfigModel):
    database_path: str = "runtime/stacmem.sqlite3"
    request_timeout_seconds: float = Field(default=300.0, gt=0)


class ScopeConfig(StrictConfigModel):
    app_id: str = "stacmem_research"
    project_id: str = "dev"


class ExtractionConfig(StrictConfigModel):
    provider: str = "qwen"
    model: str = "qwen-plus"
    temperature: float = 0.0
    max_retries: int = Field(default=4, ge=0)
    base_url: str | None = None
    api_key_env: str | None = None
    json_mode: bool = True


class EmbeddingConfig(StrictConfigModel):
    provider: str = "qwen"
    model: str = "text-embedding-v4"
    dimensions: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=10, gt=0)
    base_url: str | None = None
    api_key_env: str | None = None
    max_retries: int = Field(default=4, ge=0)
    send_dimensions: bool = True
    space_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("space_id")
    @classmethod
    def normalize_space_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("space_id must not be blank")
        return value


class RerankConfig(StrictConfigModel):
    provider: str = "dashscope"
    model: str = "gte-rerank-v2"
    batch_size: int = Field(default=10, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class RetrievalWeights(StrictConfigModel):
    semantic: float = Field(default=0.42, ge=0)
    lexical: float = Field(default=0.13, ge=0)
    temporal: float = Field(default=0.20, ge=0)
    spatial: float = Field(default=0.12, ge=0)
    validity: float = Field(default=0.08, ge=0)
    confidence: float = Field(default=0.05, ge=0)


class RetrievalConfig(StrictConfigModel):
    candidate_pool: int = Field(default=50, gt=0)
    top_k: int = Field(default=10, gt=0)
    token_budget: int = Field(default=4000, ge=32)
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)


class ConflictConfig(StrictConfigModel):
    close_corrected_transaction: bool = True
    infer_transition_end: bool = True
    infer_transition_from_evidence: bool = True
    transition_mode: str = "proposition_v2"
    temporal_relation_mode: str = "valid_interval_v2"


class TemporalGroundingConfig(StrictConfigModel):
    enabled: bool = True
    mode: str = "certificate_v2"


class AdmissionConfig(StrictConfigModel):
    enabled: bool = True
    mode: str = "proposition_v2"


class RecoveryConfig(StrictConfigModel):
    prepared_retention: Literal["until_commit", "forever"] = "until_commit"


class AppConfig(StrictConfigModel):
    predicate_schema: PredicateSchemaConfig = Field(default_factory=PredicateSchemaConfig)
    place_schema: PlaceSchemaConfig = Field(default_factory=PlaceSchemaConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    conflict: ConflictConfig = Field(default_factory=ConflictConfig)
    temporal_grounding: TemporalGroundingConfig = Field(default_factory=TemporalGroundingConfig)
    admission: AdmissionConfig = Field(default_factory=AdmissionConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        config = cls.model_validate(data)
        db_path = Path(config.runtime.database_path).expanduser()
        if not db_path.is_absolute():
            db_path = (config_path.parent.parent / db_path).resolve()
        config.runtime.database_path = str(db_path)
        config.extraction.model = os.getenv("STACMEM_LLM_MODEL", config.extraction.model)
        config.embedding.model = os.getenv("STACMEM_EMBEDDING_MODEL", config.embedding.model)
        config.embedding.dimensions = int(
            os.getenv("STACMEM_EMBEDDING_DIMENSIONS", config.embedding.dimensions)
        )
        config.rerank.model = os.getenv("STACMEM_RERANK_MODEL", config.rerank.model)
        return config
