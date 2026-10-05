"""Revalidate source-linked history into a new ledger under the current policy."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import AppConfig
from .contracts import ContractPolicies
from .models import Message, canonical_json
from .place_schema import PlaceRegistry, canonical_place_manifest
from .prepared import decode_batch, digest, policy_fingerprint
from .standalone import PROFILE, StandaloneMemory


class MigrationBlocked(ValueError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("History cannot be rebuilt completely; inspect the migration report")
        self.report = report


def _candidate(claim: dict) -> dict:
    metadata = claim.get("metadata", {})
    source = metadata.get("source_contract", {})
    temporal = metadata.get("temporal_certificate", {})
    observation = metadata.get("temporal_observation", {}).get("original", {})
    return {
        "subject": claim["subject"],
        "predicate": metadata.get("original_predicate", claim["predicate"]),
        "object_value": source.get("proposed_normalized_value", claim["object_value"]),
        "object_surface": source.get("object_surface", claim["object_value"]),
        "valid_start": observation.get("valid_start", temporal.get(
            "original_valid_start", claim.get("valid_start")
        )),
        "valid_end": observation.get("valid_end", temporal.get(
            "original_valid_end", claim.get("valid_end")
        )),
        "update_kind": observation.get("update_kind", metadata.get(
            "original_update_kind", claim.get("update_kind", "assertion")
        )),
        "confidence": claim.get("confidence", 1.0),
        "functional": claim.get("functional"),
        "source_message_ids": claim["source_message_ids"],
        "source_content": source.get("proposed_source_content", claim["source_content"]),
        "place": source.get("raw_place", claim.get("place")),
        "proposition_grounding": metadata.get("proposition_grounding", {}),
    }


def _load_history(connection, config, include_prepared):
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"source_sessions", "source_messages", "standalone_meta", "claims",
            "ledger_meta"}.issubset(tables):
        raise ValueError("Revalidation requires a source-linked standalone ledger")
    source_meta = dict(connection.execute("SELECT key,value FROM standalone_meta"))
    if source_meta.get("profile") != PROFILE:
        raise ValueError("Historical table/profile layout requires a separate format migration")
    if source_meta.get("history_migration_status") in {"processing", "failed"}:
        raise ValueError("The source is an incomplete historical rebuild")
    shared = dict(connection.execute("SELECT key,value FROM ledger_meta"))
    expected = {
        "predicate_schema": ContractPolicies(config.predicate_schema).manifest(),
        "place_schema": PlaceRegistry(config.place_schema).manifest(),
    }
    for key, value in expected.items():
        saved = [m[key] for m in (shared, source_meta) if key in m]
        if not saved:
            raise ValueError(f"Source lacks {key}; supply an explicit format/schema migration")
        for stored in saved:
            decoded = json.loads(stored)
            if key == "place_schema":
                decoded = canonical_place_manifest(decoded)
            if canonical_json(decoded) != canonical_json(value):
                raise ValueError(f"Source {key} differs from the requested revalidation config")
    for table in ("source_messages", "prepared_batches"):
        if table in tables and connection.execute(
            f"SELECT 1 FROM {table} AS item WHERE NOT EXISTS "
            "(SELECT 1 FROM source_sessions AS receipt WHERE receipt.owner_id=item.owner_id "
            "AND receipt.session_id=item.session_id) LIMIT 1"
        ).fetchone():
            raise ValueError(f"Orphaned {table} require manual provenance migration")
    space = shared.get("embedding_space")
    sessions, blocked = [], []
    receipts = connection.execute(
        "SELECT * FROM source_sessions ORDER BY received_at,rowid"
    ).fetchall()
    linked_ids = set()
    for receipt in receipts:
        previous_policy = None
        owner, session = receipt["owner_id"], receipt["session_id"]
        request = json.loads(receipt["request_json"])
        if digest(receipt["request_json"]) != receipt["fingerprint"]:
            raise ValueError(f"Saved source request checksum mismatch: {owner}/{session}")
        if request.get("owner_id") != owner or request.get("session_id") != session:
            raise ValueError("Source request identity differs from its receipt")
        messages = [Message.model_validate(m) for m in request["messages"]]
        if not messages:
            raise ValueError("A source receipt has no original messages")
        actual = {m.message_id: m for m in messages}
        if len(actual) != len(messages) or None in actual:
            raise ValueError("Source request requires unique real message IDs")
        persisted = connection.execute(
            "SELECT message_id,timestamp,role,content FROM source_messages "
            "WHERE owner_id=? AND session_id=?", (owner, session)
        ).fetchall()
        if len(persisted) != len(actual) or any(
            row["message_id"] not in actual or
            (row["timestamp"], row["role"], row["content"]) != (
                actual[row["message_id"]].timestamp,
                actual[row["message_id"]].role,
                actual[row["message_id"]].text(),
            ) for row in persisted
        ):
            raise ValueError("Persisted messages differ from the original source request")
        old = connection.execute(
            "SELECT id,status,valid_start,valid_end,update_kind,metadata_json FROM claims "
            "WHERE owner_id=? AND source_session_id=?", (owner, session)
        ).fetchall()
        if receipt["status"] == "committed":
            response = json.loads(receipt["response_json"])
            candidates = response["claims"]
            old_by_id = {}
            for row in old:
                state = dict(row)
                state["temporal_eligibility"] = json.loads(
                    state.pop("metadata_json")
                ).get("temporal_eligibility")
                old_by_id[row["id"]] = state
            ids = [c["id"] for c in candidates]
            if len(set(ids)) != len(ids) or set(ids) != set(old_by_id):
                raise ValueError("Committed receipt and historical claims do not match")
            if response.get("owner_id") != owner or response.get("session_id") != session:
                raise ValueError("Committed response identity differs from its receipt")
            old_states = [old_by_id[c["id"]] for c in candidates]
            linked_ids.update(ids)
        else:
            prepared = connection.execute(
                "SELECT * FROM prepared_batches WHERE owner_id=? AND session_id=?",
                (owner, session),
            ).fetchone() if "prepared_batches" in tables else None
            reason = None
            if not include_prepared:
                reason = "incomplete_session_requires_explicit_include_prepared"
            elif receipt["status"] not in {"failed", "prepared"}:
                reason = "processing_session_requires_writer_or_extraction_diagnosis"
            elif old:
                reason = "partial_legacy_claims_require_manual_migration"
            elif prepared is None:
                reason = "no_saved_candidates_requires_new_extraction"
            if reason:
                blocked.append({"owner_id": owner, "session_id": session, "reason": reason})
                continue
            if digest(prepared["payload_json"]) != prepared["payload_sha256"]:
                raise ValueError("Prepared source checksum mismatch")
            payload = json.loads(prepared["payload_json"])
            previous_policy = payload.get("policy_fingerprint")
            if not space or not isinstance(payload.get("policy_fingerprint"), str):
                raise ValueError("Prepared source lacks an embedding/policy identity")
            # The old policy is recorded, not applied. Only its intact proposals are imported.
            drafts, _ = decode_batch(
                prepared["payload_json"], owner_id=owner, session_id=session,
                request_fingerprint=receipt["fingerprint"],
                policy=payload["policy_fingerprint"], embedding_space=digest(space),
            )
            candidates = [d.model_dump(mode="json") for d in drafts]
            old_states = [None] * len(candidates)
        if any(c.get("owner_id") != owner or c.get("source_session_id") != session
               for c in candidates):
            raise ValueError("A saved candidate belongs to another source receipt")
        sessions.append({
            "owner_id": owner, "session_id": session, "messages": messages,
            "cached_claims": {"claims": [_candidate(c) for c in candidates]},
            "old_states": old_states, "source_status": receipt["status"],
            "previous_prepared_policy": previous_policy,
        })
    unlinked = [r[0] for r in connection.execute("SELECT id FROM claims") if r[0] not in linked_ids]
    if unlinked and not blocked:
        raise ValueError("Claims without committed source receipts require manual migration")
    return sessions, blocked


def _relation_report(source, rebuilt, new_to_old: dict[str, str]) -> dict[str, Any]:
    before_rows = source.execute(
        "SELECT source_claim_id,target_claim_id,relation FROM conflict_relations"
    ).fetchall()
    after_rows = rebuilt.execute(
        "SELECT source_claim_id,target_claim_id,relation FROM conflict_relations"
    ).fetchall()
    before_counts = Counter(row["relation"] for row in before_rows)
    after_counts = Counter(row["relation"] for row in after_rows)

    def graph(rows, *, translate):
        edges: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            pair = (translate(row["source_claim_id"]), translate(row["target_claim_id"]))
            edges.setdefault(pair, set()).add(row["relation"])
        return edges

    before = graph(before_rows, translate=lambda claim_id: claim_id)
    after = graph(after_rows, translate=lambda claim_id: new_to_old.get(claim_id, claim_id))
    changes: dict[str, list[dict[str, Any]]] = {"added": [], "removed": [], "changed": []}
    for pair in sorted(before.keys() | after.keys()):
        old_types, new_types = before.get(pair), after.get(pair)
        if old_types == new_types:
            continue
        kind = "changed" if old_types and new_types else "removed" if old_types else "added"
        changes[kind].append({
            "source_claim_id": pair[0], "target_claim_id": pair[1],
            "before": sorted(old_types or ()), "after": sorted(new_types or ()),
        })
    return {
        "relation_counts_before": dict(sorted(before_counts.items())),
        "relation_counts_after": dict(sorted(after_counts.items())),
        "relation_changes": changes,
    }


def revalidate_history(
    source: str | Path, *, config: AppConfig | None = None,
    output: str | Path | None = None, include_prepared: bool = False,
    allow_paid_api: bool = False,
) -> dict[str, Any]:
    """Audit in a temporary offline ledger, or build a separate application database.

    Reuses saved proposals and authentic original messages, not prior admission decisions.
    Output migrations rebuild vectors using the requested embedding config. Audit is offline.
    """
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve() if output is not None else None
    if destination and destination.exists():
        raise FileExistsError("Migration output must be a new directory")
    settings = (config or AppConfig()).model_copy(deep=True)
    if destination is None or config is None:
        settings.embedding.provider = "hash"
        settings.embedding.dimensions = 128
        settings.embedding.space_id = None
    if settings.embedding.provider != "hash" and not allow_paid_api:
        raise ValueError("Rebuilding online embeddings requires --allow-paid-api")
    settings.extraction.provider = "rule"
    settings.rerank.provider = "none"
    report: dict[str, Any] = {
        "kind": "historical_policy_revalidation", "source": str(source),
        "mode": "rebuild" if destination else "audit", "status": "validating",
        "new_policy_fingerprint": policy_fingerprint(settings),
        "extraction_calls": 0, "embedding_provider": settings.embedding.provider,
        "embedding_api_calls_possible": settings.embedding.provider != "hash",
        "sessions_revalidated": 0, "prepared_sessions_revalidated": 0,
        "changes": [], "source_database_modified": False,
    }
    with tempfile.TemporaryDirectory(prefix="stacmem-revalidate-") as temporary:
        snapshot_path = Path(temporary) / "source.sqlite3"
        reader = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=30)
        snapshot = sqlite3.connect(snapshot_path)
        snapshot.row_factory = sqlite3.Row
        try:
            reader.backup(snapshot)
            with snapshot_path.open("rb") as handle:
                report["source_snapshot_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            sessions, blocked = _load_history(snapshot, settings, include_prepared)
            report["blocked_sessions"] = blocked
            report["previous_prepared_policies"] = sorted({
                s["previous_prepared_policy"] for s in sessions if s["previous_prepared_policy"]
            })
            if blocked:
                report["status"] = "blocked"
                if destination:
                    raise MigrationBlocked(report)
                return report
            if destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                workspace = Path(tempfile.mkdtemp(
                    prefix=f".{destination.name}.staging-", dir=destination.parent,
                ))
            else:
                workspace = Path(temporary) / "audit"
                workspace.mkdir()
            try:
                settings.runtime.database_path = str(workspace / "memory.sqlite3")
                if destination:
                    backup = sqlite3.connect(workspace / "source_snapshot.sqlite3")
                    try:
                        snapshot.backup(backup)
                    finally:
                        backup.close()
                    # Block ordinary readers even if provider construction fails immediately.
                    with closing(sqlite3.connect(settings.runtime.database_path)) as target, target:
                        target.execute(
                            "CREATE TABLE standalone_meta (key TEXT PRIMARY KEY, "
                            "value TEXT NOT NULL)"
                        )
                        target.execute(
                            "INSERT INTO standalone_meta VALUES "
                            "('history_migration_status','processing')"
                        )
                with StandaloneMemory(settings, _migration_build=bool(destination)) as memory:
                    pairs = []
                    new_to_old = {}
                    for data in sessions:
                        result = memory.remember(
                            owner_id=data["owner_id"], session_id=data["session_id"],
                            messages=data["messages"], cached_claims=data["cached_claims"],
                        )
                        if len(result["claims"]) != len(data["old_states"]):
                            raise ValueError("Candidate count changed; preserve incomplete rebuild")
                        for claim, old in zip(result["claims"], data["old_states"], strict=True):
                            pairs.append((data, claim["id"], old))
                            if old:
                                new_to_old[claim["id"]] = old["id"]
                        report["sessions_revalidated"] += 1
                        report["prepared_sessions_revalidated"] += (
                            data["source_status"] != "committed"
                        )
                    counts: Counter = Counter()
                    for data, claim_id, old in pairs:
                        claim = memory.memory.store.get_claims([claim_id])[0]
                        counts[claim.status.value] += 1
                        current = {key: getattr(claim, key) for key in
                                   ("status", "valid_start", "valid_end", "update_kind")}
                        current["temporal_eligibility"] = claim.metadata.get("temporal_eligibility")
                        previous = {key: old[key] for key in current} if old else None
                        if previous != current:
                            report["changes"].append({
                                "owner_id": data["owner_id"], "session_id": data["session_id"],
                                "previous_claim_id": old["id"] if old else None,
                                "new_claim_id": claim.id, "predicate": claim.predicate,
                                "value": claim.object_value, "before": previous, "after": current,
                                "admission_reasons": claim.metadata.get("admission_reasons", []),
                            })
                    report.update({"status": "completed", "claim_statuses": dict(counts),
                                   "claims_revalidated": len(pairs),
                                   "changed_claims": len(report["changes"]),
                                   "embedding_space": memory.memory.store.embedding_space()})
                    report.update(_relation_report(snapshot, memory.memory.store._conn, new_to_old))
                    if destination:
                        report["output"] = str(destination)
                        with memory._db:
                            memory._db.execute(
                                "UPDATE standalone_meta SET value='completed' "
                                "WHERE key='history_migration_status'"
                            )
                        (workspace / "report.json").write_text(
                            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                if destination:
                    if destination.exists():
                        raise FileExistsError("Migration output was created during rebuild")
                    workspace.rename(destination)
            except BaseException as exc:
                if destination:
                    report.update({
                        "status": "failed", "error_type": type(exc).__name__,
                        "staging_path": str(workspace),
                    })
                    with closing(sqlite3.connect(workspace / "memory.sqlite3")) as target, target:
                        target.execute(
                            "CREATE TABLE IF NOT EXISTS standalone_meta "
                            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                        )
                        target.execute(
                            "INSERT OR REPLACE INTO standalone_meta VALUES "
                            "('history_migration_status','failed')"
                        )
                    (workspace / "report.json").write_text(
                        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                raise
        finally:
            snapshot.close()
            reader.close()
    return report
