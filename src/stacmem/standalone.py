"""Durable STAC-Mem runtime with source receipts and recovery controls.

Source inputs are preserved before extraction. Prepared claim batches and receipts
commit atomically. Incomplete owners require explicit, validated recovery.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import AppConfig
from .contracts import ContractPolicies, compile_claim_payload
from .models import EvidencePack, Message, QueryFrame, canonical_json
from .pipeline import StacMemory
from .place_schema import PlaceRegistry, canonical_place_manifest
from .prepared import PROTOCOL, decode_batch, digest, encode_batch, policy_fingerprint
from .time_utils import now_ms

PROFILE = "standalone-state-v1"
LEGACY_PROFILES = {"standalone-source-v1+contracts-v15"}


class ReceiptConflict(ValueError):
    """A session ID was reused with different inputs."""


class IncompleteWrite(RuntimeError):
    """State may be partial; inspect instead of automatically repeating writes."""


class StandaloneMemory:
    """Local trusted-user SDK. Owner IDs are filters, not authentication."""

    def __init__(
        self, config: AppConfig, *, _recovery_only: bool = False,
        _migration_build: bool = False,
    ) -> None:
        policies = ContractPolicies(config.predicate_schema)
        schema_json = canonical_json(policies.manifest())
        self._schema_fingerprint = digest(schema_json)
        places = PlaceRegistry(config.place_schema)
        self._place_registry = places
        place_schema_json = canonical_json(places.manifest())
        self._place_schema_fingerprint = digest(place_schema_json)
        self._place_schema_manifest = places.manifest()
        path = Path(config.runtime.database_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._recovery_only = _recovery_only
        try:
            tables = {
                row[0]
                for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "standalone_meta" in tables:
                migration = self._db.execute(
                    "SELECT value FROM standalone_meta WHERE key='history_migration_status'"
                ).fetchone()
                if migration and migration[0] in {"processing", "failed"} and not _migration_build:
                    raise IncompleteWrite("Historical rebuild is incomplete; inspect its report")
                stored = self._db.execute(
                    "SELECT value FROM standalone_meta WHERE key='profile'"
                ).fetchone()
                if stored and stored[0] in LEGACY_PROFILES:
                    raise ValueError(
                        "Legacy database detected; back it up and use an explicit migration "
                        "or a new database"
                    )
                if stored and stored[0] != PROFILE:
                    raise ValueError(f"Unsupported stored profile: {stored[0]}")
            if "claims" in tables and "standalone_meta" not in tables:
                self._validate_empty_engine_ledger(tables, schema_json, place_schema_json)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS standalone_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_sessions (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, request_json TEXT NOT NULL,
                    mode TEXT NOT NULL, received_at INTEGER NOT NULL,
                    completed_at INTEGER, status TEXT NOT NULL,
                    response_json TEXT, error_type TEXT,
                    PRIMARY KEY(owner_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS source_messages (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL, timestamp INTEGER NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL,
                    PRIMARY KEY(owner_id, session_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS recovery_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    started_at INTEGER NOT NULL, previous_receipt_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prepared_batches (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    prepared_at INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, session_id),
                    FOREIGN KEY(owner_id, session_id)
                        REFERENCES source_sessions(owner_id, session_id)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS source_fts USING fts5(
                    owner_id UNINDEXED, session_id UNINDEXED,
                    message_id UNINDEXED, content
                );
                """
            )
            self._db.execute(
                "INSERT OR IGNORE INTO standalone_meta VALUES ('profile', ?)", (PROFILE,)
            )
            profile = self._db.execute(
                "SELECT value FROM standalone_meta WHERE key='profile'"
            ).fetchone()[0]
            if profile != PROFILE:
                raise ValueError(f"Unsupported stored profile: {profile}")
            self._db.commit()
            self._bind_schemas(schema_json, place_schema_json)
            self.memory = StacMemory.from_app_config(
                config, _recovery_only=_recovery_only, _migration_build=_migration_build
            )
            self._policy = policy_fingerprint(config)
            self._embedding_space_fingerprint = digest(
                canonical_json(self.memory.store.embedding_space())
            )
            self._prepared_retention = config.recovery.prepared_retention
        except BaseException:
            self._db.close()
            raise

    def _bind_schemas(self, predicate_json: str, place_json: str) -> None:
        """Check both source contracts in one transaction before creating the engine."""
        try:
            self._db.execute("BEGIN IMMEDIATE")
            saved = dict(self._db.execute(
                "SELECT key,value FROM standalone_meta WHERE key IN "
                "('predicate_schema','place_schema')"
            ).fetchall())
            has_sources = self._db.execute("SELECT 1 FROM source_sessions LIMIT 1").fetchone()
            has_claim_table = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claims'"
            ).fetchone()
            has_claims = has_claim_table and self._db.execute(
                "SELECT 1 FROM claims LIMIT 1"
            ).fetchone()
            expected = {"predicate_schema": predicate_json, "place_schema": place_json}
            has_ledger_meta = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger_meta'"
            ).fetchone()
            shared = dict(self._db.execute(
                "SELECT key,value FROM ledger_meta WHERE key IN "
                "('predicate_schema','place_schema')"
            ).fetchall()) if has_ledger_meta else {}
            for key, current in expected.items():
                previous = saved.get(key)
                if previous is None and (has_sources or has_claims):
                    raise ValueError(f"Populated database lacks {key}; migrate separately")
                if previous is not None:
                    if key == "place_schema":
                        previous = canonical_json(
                            canonical_place_manifest(json.loads(previous))
                        )
                    if previous != current:
                        raise ValueError(f"{key} differs from stored schema; use a new database")
                shared_value = shared.get(key)
                if key == "place_schema" and shared_value is not None:
                    shared_value = canonical_json(
                        canonical_place_manifest(json.loads(shared_value))
                    )
                if shared_value is not None and shared_value != current:
                    raise ValueError(f"{key} differs from stored ledger; use a new database")
            for key, current in expected.items():
                if key not in saved:
                    self._db.execute(
                        "INSERT INTO standalone_meta VALUES (?, ?)", (key, current)
                    )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise

    def _validate_empty_engine_ledger(
        self, tables: set[str], predicate_json: str, place_json: str
    ) -> None:
        """Bootstrap only an empty engine ledger; direct writes have no source receipts."""
        if not {"ledger_meta", "conflict_relations", "commits", "claims_fts"}.issubset(tables):
            raise ValueError("Use a new database or migrate the existing schema before startup")
        for table in ("claims", "conflict_relations", "commits", "prepared_batches",
                      "source_sessions", "source_messages"):
            if table in tables and self._db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise ValueError(
                    "Use a new database or migrate the existing schema before startup; "
                    "only an empty direct engine ledger can be bootstrapped"
                )
        stored = dict(self._db.execute(
            "SELECT key,value FROM ledger_meta WHERE key IN "
            "('predicate_schema','place_schema','embedding_space')"
        ).fetchall())
        if not {"predicate_schema", "place_schema", "embedding_space"}.issubset(stored):
            raise ValueError("Use a new database or migrate the unbound existing schema")
        for key, current in {"predicate_schema": predicate_json,
                             "place_schema": place_json}.items():
            saved = stored.get(key)
            if key == "place_schema" and saved is not None:
                saved = canonical_json(canonical_place_manifest(json.loads(saved)))
            if saved is not None and saved != current:
                raise ValueError(f"{key} differs from stored ledger; use a new database")

    @classmethod
    def offline(cls, database: str | Path) -> StandaloneMemory:
        config = AppConfig()
        config.runtime.database_path = str(database)
        config.embedding.provider = "hash"
        config.embedding.dimensions = 128
        config.extraction.provider = "rule"
        config.rerank.provider = "none"
        return cls(config)

    @classmethod
    def recovery(cls, config: AppConfig) -> StandaloneMemory:
        """Open saved state without model credentials for prepared-batch recovery only."""
        return cls(config, _recovery_only=True)

    def close(self) -> None:
        self._db.close()
        self.memory.close()

    def __enter__(self) -> StandaloneMemory:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _assert_owner_ready(self, owner_id: str) -> None:
        row = self._db.execute(
            "SELECT session_id, status FROM source_sessions "
            "WHERE owner_id=? AND status!='committed' LIMIT 1",
            (owner_id,),
        ).fetchone()
        if row:
            raise IncompleteWrite(
                f"Owner has {row['status']} session {row['session_id']}; "
                "inspect the receipt and preserve the database before recovery"
            )

    def _require_operational(self) -> None:
        if self._recovery_only:
            raise RuntimeError(
                "This recovery-only handle permits status, inspection, source search and "
                "recover_prepared only"
            )

    def remember(
        self,
        *,
        owner_id: str,
        session_id: str,
        messages: list[Message],
        cached_claims: dict[str, Any] | None = None,
        retry_failed_empty: bool = False,
    ) -> dict[str, Any]:
        """Persist sources before extraction. Cached claims are a diagnostic input.

        Successful identical retries replay the original response, not current
        claim status. Prepared batches are committed atomically; interrupted
        owners remain blocked until explicit recovery from the saved batch.
        """
        self._require_operational()
        if not owner_id.strip() or not session_id.strip() or not messages:
            raise ValueError("Nonempty owner_id, session_id and messages are required")
        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError("This standalone entry accepts nonempty TEXT messages only")
            normalized.append(
                message.model_copy(
                    update={"message_id": message.message_id or f"{session_id}:{index}"}
                )
            )
        if len({m.message_id for m in normalized}) != len(normalized):
            raise ValueError("message_id must be unique within a session")
        if cached_claims is not None and (
            not isinstance(cached_claims, dict)
            or not isinstance(cached_claims.get("claims"), list)
            or any(not isinstance(c, dict) for c in cached_claims["claims"])
        ):
            raise ValueError("cached_claims must contain a claims list of objects")
        mode = "cached-structure-diagnostic" if cached_claims is not None else "model-extraction"
        request = canonical_json(
            {
                "profile": PROFILE,
                "owner_id": owner_id,
                "session_id": session_id,
                "messages": [m.model_dump(mode="json") for m in normalized],
                "cached_claims": cached_claims,
                "mode": mode,
            }
        )
        request_digest = hashlib.sha256(request.encode()).hexdigest()
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                old = self._db.execute(
                    "SELECT * FROM source_sessions WHERE owner_id=? AND session_id=?",
                    (owner_id, session_id),
                ).fetchone()
                if old:
                    if old["fingerprint"] != request_digest:
                        raise ReceiptConflict("Session ID already exists with different inputs")
                    if old["status"] == "committed":
                        self._db.commit()
                        return {**json.loads(old["response_json"]), "replayed": True}
                    if not retry_failed_empty or old["status"] != "failed":
                        raise IncompleteWrite(
                            "Previous write is incomplete; automatic replay blocked"
                        )
                    # The owner reservation and zero-claim check share a DB write lock.
                    count = self._db.execute(
                        "SELECT count(*) FROM claims WHERE owner_id=? AND source_session_id=?",
                        (owner_id, session_id),
                    ).fetchone()[0]
                    blocked = self._db.execute(
                        "SELECT 1 FROM source_sessions WHERE owner_id=? AND session_id!=? "
                        "AND status!='committed' LIMIT 1",
                        (owner_id, session_id),
                    ).fetchone()
                    if count or blocked:
                        raise IncompleteWrite(
                            "Partial claims or another incomplete session exist; retry refused"
                        )
                    if self._db.execute(
                        "SELECT 1 FROM prepared_batches WHERE owner_id=? AND session_id=?",
                        (owner_id, session_id),
                    ).fetchone():
                        raise IncompleteWrite("Prepared batch exists; use recover-prepared")
                else:
                    self._assert_owner_ready(owner_id)
                if cached_claims is None and self.memory.claim_extractor is None:
                    raise ValueError("Offline mode requires explicitly supplied cached_claims")
                if old:
                    self._db.execute(
                        "INSERT INTO recovery_attempts "
                        "(owner_id,session_id,started_at,previous_receipt_json) VALUES (?,?,?,?)",
                        (owner_id, session_id, now_ms(), canonical_json(dict(old))),
                    )
                    self._db.execute(
                        "UPDATE source_sessions SET status='processing',error_type=NULL "
                        "WHERE owner_id=? AND session_id=?",
                        (owner_id, session_id),
                    )
                else:
                    self._db.execute(
                        "INSERT INTO source_sessions "
                        "(owner_id,session_id,fingerprint,request_json,mode,received_at,status) "
                        "VALUES (?,?,?,?,?,?,'processing')",
                        (owner_id, session_id, request_digest, request, mode, now_ms()),
                    )
                for message in [] if old else normalized:
                    self._db.execute(
                        "INSERT INTO source_messages VALUES (?,?,?,?,?,?)",
                        (
                            owner_id,
                            session_id,
                            message.message_id,
                            message.timestamp,
                            message.role,
                            message.content,
                        ),
                    )
                    self._db.execute(
                        "INSERT INTO source_fts VALUES (?,?,?,?)",
                        (owner_id, session_id, message.message_id, message.content),
                    )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

            try:
                if cached_claims is None:
                    drafts = self.memory.claim_extractor.extract(
                        owner_id=owner_id, session_id=session_id, messages=normalized
                    )
                else:
                    drafts = compile_claim_payload(
                        cached_claims,
                        owner_id=owner_id,
                        session_id=session_id,
                        messages=normalized,
                        place_registry=self._place_registry,
                    )
                    for draft in drafts:
                        draft.extractor = "cached-structure-diagnostic"
                vectors = self.memory.embed_drafts(drafts)
                batch = encode_batch(
                    drafts, vectors, owner_id=owner_id, session_id=session_id,
                    request_fingerprint=request_digest, policy=self._policy,
                    embedding_space=self._embedding_space_fingerprint,
                )
                with self._db:
                    self._db.execute(
                        "INSERT INTO prepared_batches VALUES (?,?,?,?,?)",
                        (owner_id, session_id, batch, digest(batch), now_ms()),
                    )
                    self._db.execute(
                        "UPDATE source_sessions SET status='prepared' "
                        "WHERE owner_id=? AND session_id=?",
                        (owner_id, session_id),
                    )
                return self._commit_prepared(owner_id=owner_id, session_id=session_id)
            except BaseException as exc:
                self._mark_failed(owner_id, session_id, exc)
                raise

    def _mark_failed(self, owner_id: str, session_id: str, exc: BaseException) -> None:
        with self._db:
            self._db.execute(
                "UPDATE source_sessions SET status='failed',error_type=? "
                "WHERE owner_id=? AND session_id=? AND status!='committed'",
                (type(exc).__name__, owner_id, session_id),
            )

    def _load_prepared(self, connection, *, owner_id, session_id):
        receipt = connection.execute(
            "SELECT * FROM source_sessions WHERE owner_id=? AND session_id=?",
            (owner_id, session_id),
        ).fetchone()
        if receipt is None:
            raise ValueError("No saved receipt for that owner/session")
        if receipt["status"] == "committed":
            return receipt, None, None
        if receipt["status"] not in {"prepared", "failed"}:
            raise IncompleteWrite("No durable prepared state; writer may still be processing")
        if connection.execute(
            "SELECT 1 FROM claims WHERE owner_id=? AND source_session_id=? LIMIT 1",
            (owner_id, session_id),
        ).fetchone():
            raise IncompleteWrite("Partial claims exist; legacy recovery requires a separate audit")
        if connection.execute(
            "SELECT 1 FROM source_sessions WHERE owner_id=? AND session_id!=? "
            "AND status!='committed' LIMIT 1", (owner_id, session_id),
        ).fetchone():
            raise IncompleteWrite("Another incomplete session exists for this owner")
        row = connection.execute(
            "SELECT * FROM prepared_batches WHERE owner_id=? AND session_id=?",
            (owner_id, session_id),
        ).fetchone()
        if row is None:
            raise IncompleteWrite("No saved batch; inspect before choosing retry-empty")
        if digest(receipt["request_json"]) != receipt["fingerprint"]:
            raise ValueError("Saved request fingerprint mismatch; preserve the database")
        if digest(row["payload_json"]) != row["payload_sha256"]:
            raise ValueError("Prepared batch checksum mismatch; preserve the database")
        drafts, vectors = decode_batch(
            row["payload_json"], owner_id=owner_id, session_id=session_id,
            request_fingerprint=receipt["fingerprint"], policy=self._policy,
            embedding_space=self._embedding_space_fingerprint,
        )
        space = self.memory.store.embedding_space()
        if vectors and (space is None or len(vectors[0]) != int(space["dimensions"])):
            raise ValueError("Prepared vectors differ from the database embedding space")
        return receipt, drafts, vectors

    def _commit_prepared(self, *, owner_id: str, session_id: str) -> dict[str, Any]:
        # Claims, prior-version mutations, FTS, relations and receipt share this connection.
        with self.memory.store.transaction() as connection:
            receipt, drafts, vectors = self._load_prepared(
                connection, owner_id=owner_id, session_id=session_id
            )
            if receipt["status"] == "committed":
                return {**json.loads(receipt["response_json"]), "replayed": True}
            outcomes = self.memory.ingest_prepared(drafts, vectors)
            result = {
                "profile": PROFILE, "write_protocol": PROTOCOL,
                "owner_id": owner_id, "session_id": session_id, "mode": receipt["mode"],
                "status": "committed", "replayed": False,
                "claims": [o.claim.model_dump(mode="json") for o in outcomes],
                "relations": [r.model_dump(mode="json") for o in outcomes for r in o.relations],
                "completed_at": now_ms(),
            }
            self._finish_receipt(connection, result)
        return result

    def _finish_receipt(self, connection, result) -> None:
        connection.execute(
            "UPDATE source_sessions SET status='committed',completed_at=?,"
            "response_json=?,error_type=NULL WHERE owner_id=? AND session_id=?",
            (result["completed_at"], canonical_json(result),
             result["owner_id"], result["session_id"]),
        )
        if self._prepared_retention == "until_commit":
            connection.execute(
                "DELETE FROM prepared_batches WHERE owner_id=? AND session_id=?",
                (result["owner_id"], result["session_id"]),
            )

    def recover_prepared(self, *, owner_id: str, session_id: str) -> dict[str, Any]:
        """Explicitly replay saved drafts/vectors, without extraction or embedding calls.

        SQLite serializes competing commit/recovery attempts. A still-running
        original writer rechecks the receipt and replays the committed result.
        Processing without a prepared batch and legacy partial writes remain blocked.
        """
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                receipt, _, _ = self._load_prepared(
                    self._db, owner_id=owner_id, session_id=session_id
                )
                if receipt["status"] == "committed":
                    self._db.commit()
                    return {**json.loads(receipt["response_json"]), "replayed": True}
                self._db.execute(
                    "INSERT INTO recovery_attempts "
                    "(owner_id,session_id,started_at,previous_receipt_json) VALUES (?,?,?,?)",
                    (owner_id, session_id, now_ms(), canonical_json(dict(receipt))),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            try:
                return self._commit_prepared(owner_id=owner_id, session_id=session_id)
            except BaseException as exc:
                self._mark_failed(owner_id, session_id, exc)
                raise

    def retry_failed_empty(self, *, owner_id: str, session_id: str) -> dict[str, Any]:
        """Explicit retry from saved inputs; only failed receipts with zero claims qualify.

        May invoke paid providers again. Saved prepared batches must instead use
        recover_prepared. Processing/legacy partial batches remain blocked.
        """
        self._require_operational()
        with self._lock:
            row = self._db.execute(
                "SELECT request_json FROM source_sessions WHERE owner_id=? AND session_id=?",
                (owner_id, session_id),
            ).fetchone()
            if row is None:
                raise ValueError("No saved receipt for that owner/session")
            request = json.loads(row["request_json"])
            return self.remember(
                owner_id=owner_id,
                session_id=session_id,
                messages=[Message.model_validate(m) for m in request["messages"]],
                cached_claims=request["cached_claims"],
                retry_failed_empty=True,
            )

    def inspect_session(self, *, owner_id: str, session_id: str) -> dict[str, Any]:
        from .inspection import inspect_session

        return inspect_session(
            self.memory.store.path, owner_id=owner_id, session_id=session_id
        )

    def search(
        self, *, owner_id: str, query: str, frame: QueryFrame | None = None, top_k: int = 10
    ) -> dict[str, Any]:
        self._require_operational()
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if frame is not None and (frame.owner_id != owner_id or frame.raw_query != query):
            raise ValueError("Query frame owner and text must match the request")
        if frame is None and self.memory.claim_extractor is None:
            raise ValueError("Offline state search requires a structured QueryFrame")
        with self._lock:
            self._assert_owner_ready(owner_id)
            before = self._owner_revision(owner_id)
            frame = frame or self.memory.query_compiler.compile(owner_id=owner_id, query=query)
            if frame.owner_id != owner_id or frame.raw_query != query:
                raise ValueError("Compiled query frame owner and text must match the request")
            if frame.temporal_intent.value == "history":
                frame = frame.model_copy(update={"expected_cardinality": "many"})
            policies = self.memory.conflict.policies
            target = (
                policies.canonicalize(frame.target_predicate)
                if frame.target_predicate
                else None
            )
            frame = frame.model_copy(update={"target_predicate": target})
            if target not in policies.specs:
                result = EvidencePack(
                    query=frame, claims=[],
                    warnings=[
                        "unsupported state predicate; use source_search or clarify the query"
                    ],
                    context="No supported state target. Original sources are available separately.",
                    diagnostics={"query_routing": "unsupported_predicate",
                                 "selected_claims": 0},
                ).model_dump(mode="json")
            else:
                result = self.memory.search(
                    owner_id=owner_id, query=query, frame=frame, top_k=top_k
                ).model_dump(mode="json")
            # Detect overlapping writes from another instance; never label a mixed read stable.
            self._assert_owner_ready(owner_id)
            if self._owner_revision(owner_id) != before:
                raise IncompleteWrite("Owner changed during search; retry the read")
            result["diagnostics"]["profile"] = PROFILE
            result["diagnostics"]["runtime_profile"] = PROFILE
            result["diagnostics"]["schema_fingerprint"] = self._schema_fingerprint
            return result

    def _owner_revision(self, owner_id: str) -> tuple:
        return tuple(
            self._db.execute(
                "SELECT session_id,status FROM source_sessions "
                "WHERE owner_id=? ORDER BY session_id",
                (owner_id,),
            ).fetchall()
        )

    def ask(
        self,
        *,
        owner_id: str,
        query: str,
        backend,
        frame: QueryFrame | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Answer a state question from the same evidence returned by search."""
        self._require_operational()
        from .answering import answer_evidence

        evidence = self.search(owner_id=owner_id, query=query, frame=frame, top_k=top_k)
        return answer_evidence(evidence, backend)

    def source_search(self, *, owner_id: str, query: str, top_k: int = 10) -> list[dict]:
        """Literal FTS retrieval, NOT a state answer or semantic entailment check."""
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        terms = list(dict.fromkeys(re.findall(r"\w+", query)))[:32]
        if not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        with self._lock:
            rows = self._db.execute(
                "SELECT f.session_id,f.message_id,f.content,s.status AS extraction_status "
                "FROM source_fts f JOIN source_sessions s "
                "ON s.owner_id=f.owner_id AND s.session_id=f.session_id "
                "WHERE source_fts MATCH ? AND f.owner_id=? ORDER BY bm25(source_fts) LIMIT ?",
                (expression, owner_id, top_k),
            ).fetchall()
            return [dict(row) for row in rows]

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(
                self._db.execute(
                    "SELECT status,count(*) FROM source_sessions GROUP BY status"
                ).fetchall()
            )
            return {
                "profile": PROFILE,
                "write_protocol": PROTOCOL,
                "predicate_schema": {
                    "fingerprint": self._schema_fingerprint,
                    "manifest": self.memory.conflict.policies.manifest(),
                },
                "place_schema": {
                    "fingerprint": self._place_schema_fingerprint,
                    "manifest": self._place_schema_manifest,
                },
                "embedding_space": self.memory.store.embedding_space(),
                "sessions": counts,
                "state_operations_blocked_for_incomplete_owners": True,
                "recovery": {
                    "prepared_retention": self._prepared_retention,
                    "prepared_batches": self._db.execute(
                        "SELECT count(*) FROM prepared_batches"
                    ).fetchone()[0],
                    "failed_empty_supported": True,
                    "prepared_batches_supported": True,
                    "new_session_atomic_commit": True,
                    "partial_batches_supported": False,
                    "attempts": self._db.execute(
                        "SELECT count(*) FROM recovery_attempts"
                    ).fetchone()[0],
                },
                "messages": self._db.execute("SELECT count(*) FROM source_messages").fetchone()[0],
                "ledger": self.memory.store.stats(),
            }
