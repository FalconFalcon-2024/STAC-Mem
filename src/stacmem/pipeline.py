"""End-to-end STAC-Mem ingestion and retrieval orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .admission import FactualityAdmissionGate
from .config import AppConfig
from .conflict import ConflictEngine, ConflictOutcome
from .contracts import ContractPolicies
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .extractors.base import ClaimExtractor, QueryCompiler
from .extractors.qwen import QwenClaimExtractor, QwenQueryCompiler
from .extractors.rules import RuleQueryCompiler
from .models import ClaimDraft, CommitWatermark, EvidencePack, Message, QueryFrame, canonical_json
from .place_schema import PlaceRegistry
from .providers import build_json_backend, endpoint_options
from .rerank import DashScopeReranker
from .resolver import StateResolver
from .retrieval import VARIANTS, ClaimRetriever
from .store import ClaimStore
from .temporal_grounding import TemporalGrounder
from .time_utils import now_ms


class StacMemory:
    def __init__(
        self,
        *,
        config: AppConfig,
        store: ClaimStore,
        conflict: ConflictEngine,
        retriever: ClaimRetriever,
        resolver: StateResolver,
        query_compiler: QueryCompiler,
        claim_extractor: ClaimExtractor | None,
    ) -> None:
        self.config = config
        self.store = store
        self.conflict = conflict
        self.retriever = retriever
        self.resolver = resolver
        self.query_compiler = query_compiler
        self.claim_extractor = claim_extractor
        self.embedder = retriever.embedder
        self.place_registry = None

    @classmethod
    def from_config(cls, path: str | Path) -> StacMemory:
        config = AppConfig.load(path)
        return cls.from_app_config(config)

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        *,
        _recovery_only: bool = False,
        _install_contracts: bool = True,
        _migration_build: bool = False,
    ) -> StacMemory:
        if _install_contracts or _recovery_only:
            from .state_runtime import validate_contract_config

            validate_contract_config(config)
        store = ClaimStore(config.runtime.database_path)
        try:
            if _install_contracts or _recovery_only:
                store.bind_state_schemas(
                    ContractPolicies(config.predicate_schema).manifest(),
                    PlaceRegistry(config.place_schema).manifest(),
                    _migration_build=_migration_build,
                )
        except BaseException:
            store.close()
            raise
        if _recovery_only:
            try:
                space = store.embedding_space()
                if space is None:
                    raise ValueError("Recovery requires a database-bound embedding space")
                memory = cls(
                    config=config,
                    store=store,
                    conflict=ConflictEngine(store),
                    retriever=ClaimRetriever(
                        store,
                        HashingEmbedder(int(space["dimensions"])),
                        config.retrieval,
                        None,
                    ),
                    resolver=StateResolver(store, token_budget=config.retrieval.token_budget),
                    query_compiler=RuleQueryCompiler(),
                    claim_extractor=None,
                )
                from .state_runtime import install_contracts

                return install_contracts(memory)
            except BaseException:
                store.close()
                raise
        resources: list[Any] = []
        try:
            store.bind_embedding_space(embedding_space_manifest(config))
            if config.embedding.provider == "hash":
                embedder = HashingEmbedder(config.embedding.dimensions)
            elif config.embedding.provider in {"qwen", "openai_compatible"}:
                embedder = OpenAICompatibleEmbedder(
                    model=config.embedding.model,
                    dimensions=config.embedding.dimensions,
                    batch_size=config.embedding.batch_size,
                    **endpoint_options(
                        config.embedding.provider,
                        config.embedding.base_url,
                        config.embedding.api_key_env,
                    ),
                    timeout_seconds=config.runtime.request_timeout_seconds,
                    max_retries=config.embedding.max_retries,
                    send_dimensions=config.embedding.send_dimensions,
                )
            else:
                raise ValueError(f"unsupported embedding provider: {config.embedding.provider}")
            resources.append(embedder)

            if config.extraction.provider in {"qwen", "openai_compatible"}:
                claim_backend = build_json_backend(config)
                resources.append(claim_backend)
                query_backend = build_json_backend(config)
                resources.append(query_backend)
                claim_extractor: ClaimExtractor | None = QwenClaimExtractor(
                    backend=claim_backend,
                )
                query_compiler: QueryCompiler = QwenQueryCompiler(
                    backend=query_backend,
                )
                if config.extraction.provider == "openai_compatible":
                    claim_extractor.name = "openai-compatible-claim-v1"
                    query_compiler.name = "openai-compatible-query-v1"
            elif config.extraction.provider == "rule":
                claim_extractor = None
                query_compiler = RuleQueryCompiler()
            else:
                raise ValueError(f"unsupported extraction provider: {config.extraction.provider}")

            if config.rerank.provider == "dashscope":
                reranker = DashScopeReranker(
                    model=config.rerank.model,
                    batch_size=config.rerank.batch_size,
                    timeout_seconds=config.rerank.timeout_seconds,
                    max_retries=config.rerank.max_retries,
                )
                resources.append(reranker)
            elif config.rerank.provider == "none":
                reranker = None
            else:
                raise ValueError(f"unsupported rerank provider: {config.rerank.provider}")

            memory = cls(
                config=config,
                store=store,
                conflict=ConflictEngine(
                    store,
                    close_corrected_transaction=config.conflict.close_corrected_transaction,
                    infer_transition_end=config.conflict.infer_transition_end,
                    infer_transition_from_evidence=config.conflict.infer_transition_from_evidence,
                    transition_mode=config.conflict.transition_mode,
                    temporal_relation_mode=config.conflict.temporal_relation_mode,
                    temporal_grounding_mode=(
                        config.temporal_grounding.mode
                        if config.temporal_grounding.enabled
                        else "off"
                    ),
                    temporal_grounder=TemporalGrounder(),
                    admission_gate=FactualityAdmissionGate(
                        enabled=config.admission.enabled,
                        mode=config.admission.mode,
                    ),
                ),
                retriever=ClaimRetriever(store, embedder, config.retrieval, reranker),
                resolver=StateResolver(store, token_budget=config.retrieval.token_budget),
                query_compiler=query_compiler,
                claim_extractor=claim_extractor,
            )
            if _install_contracts:
                from .state_runtime import install_contracts

                return install_contracts(memory)
            return memory
        except BaseException:
            closed: set[int] = set()
            for resource in reversed(resources):
                close = getattr(resource, "close", None)
                if callable(close) and id(resource) not in closed:
                    close()
                    closed.add(id(resource))
            store.close()
            raise

    def close(self) -> None:
        self.store.close()
        closed = set()
        for resource in (
            self.retriever.reranker, self.embedder,
            getattr(self.claim_extractor, "backend", None),
            getattr(self.query_compiler, "backend", None),
        ):
            close = getattr(resource, "close", None)
            if callable(close) and id(resource) not in closed:
                close()
                closed.add(id(resource))

    def __enter__(self) -> StacMemory:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def ingest_drafts(self, drafts: list[ClaimDraft]) -> list[ConflictOutcome]:
        if self.place_registry is not None:
            drafts = [
                draft.model_copy(update={"place": self.place_registry.resolve(draft.place)})
                for draft in drafts
            ]
        return self.ingest_prepared(drafts, self.embed_drafts(drafts))

    def embed_drafts(self, drafts: list[ClaimDraft]) -> list[list[float]]:
        if not drafts:
            return []
        return self.embedder.embed(
            [
                " ".join(
                    part
                    for part in [
                        draft.subject,
                        draft.predicate,
                        draft.object_value,
                        draft.place.name if draft.place and draft.place.name else "",
                    ]
                    if part
                )
                for draft in drafts
            ]
        )
    def ingest_prepared(
        self, drafts: list[ClaimDraft], vectors: list[list[float]]
    ) -> list[ConflictOutcome]:
        if len(drafts) != len(vectors):
            raise ValueError("Prepared vector count differs from the claim count")
        with self.store.transaction():
            outcomes = [
                self.conflict.ingest(draft, vector=vector)
                for draft, vector in zip(drafts, vectors, strict=True)
            ]
            refreshed = {
                claim.id: claim
                for claim in self.store.get_claims(item.claim.id for item in outcomes)
            }
            for outcome in outcomes:
                outcome.claim = refreshed.get(outcome.claim.id, outcome.claim)
        return outcomes

    def remember_session(
        self,
        *,
        owner_id: str,
        session_id: str,
        messages: list[Message],
        app_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[CommitWatermark, list[ConflictOutcome]]:
        app = app_id or self.config.scope.app_id
        project = project_id or self.config.scope.project_id
        commit = CommitWatermark(
            session_id=session_id,
            owner_id=owner_id,
            app_id=app,
            project_id=project,
        )
        self.store.save_commit(commit)
        try:
            if self.claim_extractor is None:
                raise RuntimeError(
                    "natural-language ingestion needs the qwen extractor; "
                    "use ingest_drafts when supplying prestructured claims"
                )
            drafts = self.claim_extractor.extract(
                owner_id=owner_id,
                session_id=session_id,
                messages=messages,
            )
            commit.extraction_completed_at = now_ms()
            outcomes = self.ingest_drafts(drafts)
            commit.ledger_committed_at = now_ms()
            commit.search_visible_at = commit.ledger_committed_at
            commit.details["claim_count"] = len(outcomes)
            commit.details["quarantined_claim_count"] = sum(
                item.claim.status.value == "quarantined" for item in outcomes
            )

            commit.status = "committed"
            self.store.save_commit(commit)
            return commit, outcomes
        except Exception as exc:
            commit.status = "partial_failure"
            commit.details["error"] = f"{type(exc).__name__}: {exc}"
            self.store.save_commit(commit)
            raise

    def search(
        self,
        *,
        owner_id: str,
        query: str,
        frame: QueryFrame | None = None,
        variant: str = "full",
        top_k: int | None = None,
    ) -> EvidencePack:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        query_frame = frame or self.query_compiler.compile(owner_id=owner_id, query=query)
        if self.place_registry is not None:
            query_frame = query_frame.model_copy(
                update={"place": self.place_registry.resolve(query_frame.place)}
            )
        if query_frame.target_predicate:
            query_frame.target_predicate = self.conflict.policies.canonicalize(
                query_frame.target_predicate
            )
        method = VARIANTS[variant]
        candidates = self.retriever.retrieve(query_frame, variant=method, top_k=top_k)
        return self.resolver.resolve(
            query_frame,
            candidates,
            variant=method,
            top_k=top_k or self.config.retrieval.top_k,
        )

    def doctor(self) -> dict[str, Any]:
        return {
            "database": self.store.stats(),
            "embedding_space": self.store.embedding_space(),
            "embedding_provider": type(self.embedder).__name__,
            "claim_extractor": type(self.claim_extractor).__name__
            if self.claim_extractor
            else None,
        }


def embedding_space_manifest(config: AppConfig) -> dict[str, Any]:
    """Return a secret-free identity for vectors that may be compared in one ledger."""
    embedding = config.embedding
    if embedding.provider == "hash":
        identity = {
            "provider": "hash",
            "model": "hashing-blake2b-token-bigram-v1",
            "dimensions": embedding.dimensions,
            "send_dimensions": False,
            "endpoint_sha256": None,
        }
    elif embedding.provider in {"qwen", "openai_compatible"}:
        options = endpoint_options(
            embedding.provider, embedding.base_url, embedding.api_key_env
        )
        endpoint = options["base_url"].rstrip("/")
        identity = {
            "provider": embedding.provider,
            "model": embedding.model,
            "dimensions": embedding.dimensions,
            "send_dimensions": embedding.send_dimensions,
            "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        }
    else:
        raise ValueError(f"unsupported embedding provider: {embedding.provider}")
    automatic = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    return {**identity, "space_id": embedding.space_id or f"auto:{automatic}"}
