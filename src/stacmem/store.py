"""SQLite claim ledger and append-only conflict relation store."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    Claim,
    ClaimStatus,
    CommitWatermark,
    ConflictRelation,
    Place,
    canonical_json,
)
from .place_schema import canonical_place_manifest

_FTS_TERM = re.compile(r"[\w\-]+", re.UNICODE)


class ClaimStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()

    @contextmanager
    def transaction(self):
        """Serialize writes; nested operations must not commit their caller's batch."""
        with self._lock:
            nested = self._conn.in_transaction
            savepoint = f"batch_{uuid.uuid4().hex}"
            self._conn.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
            try:
                yield self._conn
                if nested:
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                else:
                    self._conn.commit()
            except BaseException:
                if nested:
                    self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                else:
                    self._conn.rollback()
                raise

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_value TEXT NOT NULL,
                    object_norm TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    assertion_time INTEGER NOT NULL,
                    observed_at INTEGER,
                    valid_start INTEGER,
                    valid_end INTEGER,
                    transaction_start INTEGER NOT NULL,
                    transaction_end INTEGER,
                    place_json TEXT,
                    confidence REAL NOT NULL,
                    update_kind TEXT NOT NULL,
                    functional INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source_session_id TEXT,
                    source_message_ids_json TEXT NOT NULL,
                    source_content TEXT NOT NULL,
                    extractor TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector BLOB,
                    vector_dimensions INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_claim_owner_slot
                    ON claims(owner_id, slot_key, version);
                CREATE INDEX IF NOT EXISTS idx_claim_owner_status
                    ON claims(owner_id, status, transaction_start);
                CREATE INDEX IF NOT EXISTS idx_claim_valid_time
                    ON claims(owner_id, valid_start, valid_end);

                CREATE TABLE IF NOT EXISTS conflict_relations (
                    id TEXT PRIMARY KEY,
                    source_claim_id TEXT NOT NULL REFERENCES claims(id),
                    target_claim_id TEXT NOT NULL REFERENCES claims(id),
                    relation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    detector TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(source_claim_id, target_claim_id, relation)
                );
                CREATE INDEX IF NOT EXISTS idx_relation_source
                    ON conflict_relations(source_claim_id);
                CREATE INDEX IF NOT EXISTS idx_relation_target
                    ON conflict_relations(target_claim_id);

                CREATE TABLE IF NOT EXISTS commits (
                    commit_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    app_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    received_at INTEGER NOT NULL,
                    extraction_completed_at INTEGER,
                    ledger_committed_at INTEGER,
                    search_visible_at INTEGER,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(claims)").fetchall()
            }
            if "observed_at" not in columns:
                self._conn.execute("ALTER TABLE claims ADD COLUMN observed_at INTEGER")
                self._conn.execute(
                    "UPDATE claims SET observed_at = transaction_start WHERE observed_at IS NULL"
                )
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                    claim_id UNINDEXED,
                    owner_id UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
                """
            )

    def bind_embedding_space(self, manifest: dict[str, Any]) -> None:
        """Bind one vector space to the ledger before semantic writes or reads."""
        encoded = canonical_json(manifest)
        with self.transaction() as connection:
            has_claims = connection.execute("SELECT 1 FROM claims LIMIT 1").fetchone()
            has_prepared_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prepared_batches'"
            ).fetchone()
            has_prepared = has_prepared_table and connection.execute(
                "SELECT 1 FROM prepared_batches LIMIT 1"
            ).fetchone()
            has_vector_dependent_state = bool(has_claims or has_prepared)
            row = connection.execute(
                "SELECT value FROM ledger_meta WHERE key='embedding_space'"
            ).fetchone()
            if row is None:
                if has_vector_dependent_state:
                    raise ValueError(
                        "Existing claims or prepared batches have no embedding-space binding; "
                        "use an explicit re-embedding migration or a new database"
                    )
                connection.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES ('embedding_space', ?)",
                    (encoded,),
                )
            elif row["value"] != encoded:
                if has_vector_dependent_state:
                    raise ValueError(
                        "Embedding space differs from stored database; use a new database or "
                        "an explicit re-embedding migration"
                    )
                connection.execute(
                    "UPDATE ledger_meta SET value=? WHERE key='embedding_space'", (encoded,)
                )

    def embedding_space(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM ledger_meta WHERE key='embedding_space'"
            ).fetchone()
        return json.loads(row["value"]) if row else None

    def bind_state_schemas(
        self, predicate_manifest: dict, place_manifest: dict, *,
        _migration_build: bool = False,
    ) -> None:
        """Bind both public state contracts before interpreting or changing stored claims."""
        expected = {
            "predicate_schema": canonical_json(predicate_manifest),
            "place_schema": canonical_json(canonical_place_manifest(place_manifest)),
        }
        with self.transaction() as connection:
            legacy_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='standalone_meta'"
            ).fetchone()
            legacy = {}
            if legacy_table:
                migration = connection.execute(
                    "SELECT value FROM standalone_meta WHERE key='history_migration_status'"
                ).fetchone()
                if migration and migration[0] in {"processing", "failed"} and not _migration_build:
                    raise ValueError("Historical rebuild is incomplete; inspect its report")
                legacy = dict(connection.execute(
                    "SELECT key,value FROM standalone_meta WHERE key IN "
                    "('predicate_schema','place_schema')"
                ).fetchall())
            stored = dict(connection.execute(
                "SELECT key,value FROM ledger_meta WHERE key IN "
                "('predicate_schema','place_schema')"
            ).fetchall())
            has_claims = connection.execute("SELECT 1 FROM claims LIMIT 1").fetchone()
            has_source_sessions = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_sessions'"
            ).fetchone()
            has_sources = has_source_sessions and connection.execute(
                "SELECT 1 FROM source_sessions LIMIT 1"
            ).fetchone()

            def equivalent(key: str, saved: str, current: str) -> bool:
                if key == "place_schema":
                    return canonical_json(canonical_place_manifest(json.loads(saved))) == current
                return saved == current

            for key, current in expected.items():
                previous = stored.get(key)
                if previous is not None and not equivalent(key, previous, current):
                    raise ValueError(f"{key} differs from stored database; use a new database")
                legacy_value = legacy.get(key)
                if legacy_value is not None and not equivalent(key, legacy_value, current):
                    raise ValueError(
                        f"{key} differs from stored source contract; use a new database"
                    )
                if previous is None and legacy_value is None and (has_claims or has_sources):
                    raise ValueError(
                        f"Populated database has no {key} binding; use an explicit migration"
                    )
            for key, current in expected.items():
                if key not in stored:
                    connection.execute(
                        "INSERT INTO ledger_meta(key,value) VALUES (?,?)", (key, current)
                    )

    def next_version(self, owner_id: str, slot_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS value FROM claims "
                "WHERE owner_id = ? AND slot_key = ?",
                (owner_id, slot_key),
            ).fetchone()
        return int(row["value"]) + 1

    def apply_ingest(
        self,
        claim: Claim,
        relations: Iterable[ConflictRelation],
        mutations: Iterable[dict[str, Any]],
    ) -> None:
        relation_list = list(relations)
        mutation_list = list(mutations)
        with self.transaction():
            self._insert_claim(self._conn, claim)
            for mutation in mutation_list:
                self._update_claim(self._conn, **mutation)
            for relation in relation_list:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO conflict_relations (
                        id, source_claim_id, target_claim_id, relation,
                        confidence, reason, detector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation.id,
                        relation.source_claim_id,
                        relation.target_claim_id,
                        relation.relation.value,
                        relation.confidence,
                        relation.reason,
                        relation.detector,
                        relation.created_at,
                    ),
                )

    def _insert_claim(self, conn: sqlite3.Connection, claim: Claim) -> None:
        vector_blob = None
        vector_dimensions = None
        if claim.vector is not None:
            array = np.asarray(claim.vector, dtype=np.float32)
            vector_blob = array.tobytes()
            vector_dimensions = int(array.size)
        conn.execute(
            """
            INSERT INTO claims (
                id, owner_id, subject, predicate, object_value, object_norm,
                slot_key, assertion_time, observed_at, valid_start, valid_end,
                transaction_start, transaction_end, place_json, confidence,
                update_kind, functional, status, version, source_session_id,
                source_message_ids_json, source_content, extractor,
                metadata_json, vector, vector_dimensions
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                claim.id,
                claim.owner_id,
                claim.subject,
                claim.predicate,
                claim.object_value,
                claim.object_norm,
                claim.slot_key,
                claim.assertion_time,
                claim.observed_at,
                claim.valid_start,
                claim.valid_end,
                claim.transaction_start,
                claim.transaction_end,
                canonical_json(claim.place.model_dump()) if claim.place else None,
                claim.confidence,
                claim.update_kind.value,
                int(bool(claim.functional)),
                claim.status.value,
                claim.version,
                claim.source_session_id,
                canonical_json(claim.source_message_ids),
                claim.source_content,
                claim.extractor,
                canonical_json(claim.metadata),
                vector_blob,
                vector_dimensions,
            ),
        )
        fts_content = " ".join(
            [claim.subject, claim.predicate, claim.object_value, claim.source_content]
        )
        conn.execute(
            "INSERT INTO claims_fts(claim_id, owner_id, content) VALUES (?, ?, ?)",
            (claim.id, claim.owner_id, fts_content),
        )

    @staticmethod
    def _update_claim(
        conn: sqlite3.Connection,
        *,
        claim_id: str,
        status: ClaimStatus | str | None = None,
        transaction_end: int | None = None,
        valid_end: int | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[Any] = []
        if status is not None:
            assignments.append("status = ?")
            values.append(status.value if isinstance(status, ClaimStatus) else status)
        if transaction_end is not None:
            assignments.append("transaction_end = ?")
            values.append(transaction_end)
        if valid_end is not None:
            assignments.append("valid_end = ?")
            values.append(valid_end)
        if not assignments:
            return
        values.append(claim_id)
        conn.execute(
            f"UPDATE claims SET {', '.join(assignments)} WHERE id = ?",
            values,
        )

    def slot_claims(self, owner_id: str, slot_key: str) -> list[Claim]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM claims WHERE owner_id = ? AND slot_key = ? ORDER BY version ASC",
                (owner_id, slot_key),
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def owner_claims(self, owner_id: str, *, limit: int | None = None) -> list[Claim]:
        sql = "SELECT * FROM claims WHERE owner_id = ? ORDER BY transaction_start DESC"
        params: list[Any] = [owner_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def get_claims(self, claim_ids: Iterable[str]) -> list[Claim]:
        ids = list(dict.fromkeys(claim_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM claims WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        by_id = {row["id"]: self._row_to_claim(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def lexical_search(self, owner_id: str, query: str, *, limit: int) -> list[tuple[str, float]]:
        terms = _FTS_TERM.findall(query.casefold())
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:24])
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT claim_id, bm25(claims_fts) AS rank
                    FROM claims_fts
                    WHERE claims_fts MATCH ? AND owner_id = ?
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (expression, owner_id, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(row["claim_id"]), max(0.0, -float(row["rank"]))) for row in rows]

    def relations_for(self, claim_ids: Iterable[str]) -> list[ConflictRelation]:
        ids = list(dict.fromkeys(claim_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        params = [*ids, *ids]
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM conflict_relations
                WHERE source_claim_id IN ({placeholders})
                   OR target_claim_id IN ({placeholders})
                ORDER BY created_at ASC
                """,
                params,
            ).fetchall()
        return [ConflictRelation.model_validate(dict(row)) for row in rows]

    def save_commit(self, commit: CommitWatermark) -> None:
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO commits (
                    commit_id, session_id, owner_id, app_id, project_id,
                    received_at, extraction_completed_at,
                    ledger_committed_at, search_visible_at,
                    status, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(commit_id) DO UPDATE SET
                    extraction_completed_at=excluded.extraction_completed_at,
                    ledger_committed_at=excluded.ledger_committed_at,
                    search_visible_at=excluded.search_visible_at,
                    status=excluded.status,
                    details_json=excluded.details_json
                """,
                (
                    commit.commit_id,
                    commit.session_id,
                    commit.owner_id,
                    commit.app_id,
                    commit.project_id,
                    commit.received_at,
                    commit.extraction_completed_at,
                    commit.ledger_committed_at,
                    commit.search_visible_at,
                    commit.status,
                    canonical_json(commit.details),
                ),
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                table: int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("claims", "conflict_relations", "commits")
            }
            statuses = {
                row["status"]: int(row["count"])
                for row in self._conn.execute(
                    "SELECT status, COUNT(*) AS count FROM claims GROUP BY status"
                ).fetchall()
            }
        database_files = [
            item
            for item in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm"))
            if item.exists()
        ]
        return {
            "path": str(self.path),
            "bytes": sum(item.stat().st_size for item in database_files),
            "files": {item.name: item.stat().st_size for item in database_files},
            "counts": counts,
            "claim_statuses": statuses,
        }

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        vector = None
        if row["vector"] is not None:
            vector = np.frombuffer(row["vector"], dtype=np.float32).tolist()
        place = Place.model_validate(json.loads(row["place_json"])) if row["place_json"] else None
        return Claim(
            id=row["id"],
            owner_id=row["owner_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            object_norm=row["object_norm"],
            slot_key_value=row["slot_key"],
            assertion_time=row["assertion_time"],
            observed_at=row["observed_at"],
            valid_start=row["valid_start"],
            valid_end=row["valid_end"],
            transaction_start=row["transaction_start"],
            transaction_end=row["transaction_end"],
            place=place,
            confidence=row["confidence"],
            update_kind=row["update_kind"],
            functional=bool(row["functional"]),
            status=row["status"],
            version=row["version"],
            source_session_id=row["source_session_id"],
            source_message_ids=json.loads(row["source_message_ids_json"]),
            source_content=row["source_content"],
            extractor=row["extractor"],
            metadata=json.loads(row["metadata_json"]),
            vector=vector,
        )
