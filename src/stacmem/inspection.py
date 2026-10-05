"""Read-only receipt diagnostics; never instantiate providers or repair state."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def inspect_session(database: str | Path, *, owner_id: str, session_id: str) -> dict[str, Any]:
    path = Path(database).expanduser().resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        row = connection.execute(
            "SELECT owner_id,session_id,status,mode,received_at,completed_at,error_type "
            "FROM source_sessions WHERE owner_id=? AND session_id=?",
            (owner_id, session_id),
        ).fetchone()
        if row is None:
            raise ValueError("No saved receipt for that owner/session")
        claims = connection.execute(
            "SELECT id,predicate,status,version FROM claims "
            "WHERE owner_id=? AND source_session_id=? ORDER BY predicate,version,id",
            (owner_id, session_id),
        ).fetchall()
        blocked = connection.execute(
            "SELECT session_id,status FROM source_sessions WHERE owner_id=? "
            "AND status!='committed' ORDER BY session_id", (owner_id,),
        ).fetchall()
        others_blocked = any(item["session_id"] != session_id for item in blocked)
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prepared_batches'"
        ).fetchone()
        prepared = connection.execute(
            "SELECT prepared_at,payload_sha256 FROM prepared_batches "
            "WHERE owner_id=? AND session_id=?", (owner_id, session_id),
        ).fetchone() if has_table else None
        eligible = row["status"] == "failed" and not claims and not others_blocked and not prepared
        recoverable = (
            row["status"] in {"prepared", "failed"} and bool(prepared)
            and not claims and not others_blocked
        )
        if row["status"] == "committed":
            action = "replay_identical_request"
        elif row["status"] == "processing":
            action = "check_writer_liveness_do_not_replay"
        elif claims:
            action = "preserve_database_partial_recovery_not_supported"
        elif others_blocked:
            action = "inspect_other_incomplete_sessions"
        elif recoverable:
            action = "explicit_recover_prepared"
        else:
            action = "explicit_retry_empty" if eligible else "manual_inspection"
        return {
            "receipt": dict(row),
            "claims": [dict(item) for item in claims],
            "claim_count": len(claims),
            "source_message_count": connection.execute(
                "SELECT count(*) FROM source_messages WHERE owner_id=? AND session_id=?",
                (owner_id, session_id),
            ).fetchone()[0],
            "recovery_attempts": connection.execute(
                "SELECT count(*) FROM recovery_attempts WHERE owner_id=? AND session_id=?",
                (owner_id, session_id),
            ).fetchone()[0],
            "owner_blocked_by": [dict(item) for item in blocked],
            "retry_empty_eligible": eligible,
            "prepared_batch": dict(prepared) if prepared else None,
            "prepared_recovery_candidate": recoverable,
            "recommended_action": action,
            "read_only": True,
            "network_calls": 0,
            "note": (
                "Snapshot only. Recovery rechecks eligibility; processing is not proof of death."
            ),
        }
    finally:
        connection.close()
