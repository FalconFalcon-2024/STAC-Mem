"""Free mechanism demonstration. Hand-authored structures are NOT model predictions."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import Message, Place, QueryFrame
from .standalone import StandaloneMemory
from .time_utils import ensure_ms


def fixture(
    owner: str,
    session: str,
    text: str,
    value: str,
    *,
    predicate: str = "employment.organization",
    date: str = "2024-08-01",
    kind: str = "transition",
    place: str | None = None,
) -> dict:
    message = Message(
        sender_id=owner,
        role="user",
        timestamp=ensure_ms(date),
        content=text,
        message_id=f"{session}:0",
    )
    return {
        "owner_id": owner,
        "session_id": session,
        "messages": [message],
        "cached_claims": {
            "claims": [
                {
                    "subject": owner,
                    "predicate": predicate,
                    "object_value": value,
                    "object_surface": value,
                    "valid_start": date,
                    "valid_end": None,
                    "update_kind": kind,
                    "functional": True,
                    "source_content": text,
                    "source_message_ids": [message.message_id],
                    "place": {"name": place, "role": "scope"} if place else None,
                    "proposition_grounding": {
                        "evidence_span": text,
                        "subject_alignment": "aligned",
                        "factuality": "asserted",
                        "transition_entailment": kind == "transition",
                    },
                }
            ]
        },
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    checks = []
    with StandaloneMemory.offline(output / "memory.sqlite3") as memory:
        writes = [
            fixture("job", "s1", "On 2024-08-01 I joined Northwind Labs.", "Northwind Labs"),
            fixture(
                "job",
                "s2",
                "On 2024-10-15 I left Northwind Labs and joined Contoso Health.",
                "Contoso Health",
                date="2024-10-15",
            ),
            fixture(
                "scope",
                "s1",
                "Since 2024-08-01 in Seoul I prefer metro.",
                "metro",
                predicate="preference.commute_mode",
                place="Seoul",
                kind="assertion",
            ),
            fixture(
                "scope",
                "s2",
                "Since 2024-08-01 in Busan I prefer bus.",
                "bus",
                predicate="preference.commute_mode",
                place="Busan",
                kind="assertion",
            ),
            fixture(
                "scope",
                "s3",
                "On 2024-10-15 in Busan I switched from bus to taxi.",
                "taxi",
                predicate="preference.commute_mode",
                place="Busan",
                date="2024-10-15",
            ),
            fixture(
                "observation",
                "s1",
                "As of 2024-10-15 I still live in Bern.",
                "Bern",
                predicate="residence.current",
                date="2024-10-15",
                kind="assertion",
            ),
        ]
        for data in writes:
            result = memory.remember(**data)
            records.append({"kind": "write", "result": result})
        retry = memory.remember(**writes[0])
        checks.append({"name": "successful_retry_is_replayed", "passed": retry["replayed"]})
        records.append({"kind": "retry_check", "result": retry, "check": checks[-1]})
        cases = [
            ("job", "employment.organization", "2025-01-01", None, "as_of", ["Contoso Health"]),
            ("job", "employment.organization", "2024-09-01", None, "as_of", ["Northwind Labs"]),
            (
                "job",
                "employment.organization",
                None,
                None,
                "history",
                ["Contoso Health", "Northwind Labs"],
            ),
            ("scope", "preference.commute_mode", "2025-01-01", "Seoul", "as_of", ["metro"]),
            ("scope", "preference.commute_mode", "2025-01-01", "Busan", "as_of", ["taxi"]),
            ("observation", "residence.current", "2024-10-15", None, "as_of", ["Bern"]),
            ("observation", "residence.current", "2025-01-01", None, "as_of", []),
        ]
        for owner, predicate, date, place, intent, expected in cases:
            query = f"{owner} {predicate} {date or 'history'} {place or ''}"
            frame = QueryFrame(
                raw_query=query,
                owner_id=owner,
                target_subject=owner,
                target_predicate=predicate,
                temporal_intent=intent,
                query_time=ensure_ms(date) if date else None,
                place=Place(name=place) if place else None,
            )
            result = memory.search(owner_id=owner, query=query, frame=frame)
            actual = sorted(c["claim"]["object_value"] for c in result["claims"])
            check = {
                "name": query,
                "expected": sorted(expected),
                "actual": actual,
                "passed": actual == sorted(expected),
            }
            checks.append(check)
            records.append({"kind": "query", "check": check, "result": result})
        sources = memory.source_search(owner_id="job", query="Northwind")
        checks.append({"name": "original_sources_preserved", "passed": len(sources) == 2})
        records.append({"kind": "source_check", "results": sources, "check": checks[-1]})
        isolated_sources = memory.source_search(owner_id="unknown", query="Northwind")
        checks.append(
            {
                "name": "owner_isolation",
                "passed": isolated_sources == [],
            }
        )
        records.append(
            {"kind": "isolation_check", "results": isolated_sources, "check": checks[-1]}
        )
        status = memory.status()
    summary = {
        "kind": "local_lifecycle_demo",
        "input_mode": "hand_authored_cached_claims_and_query_frames",
        "network_calls": 0,
        "runtime": "stacmem",
        "checks": len(checks),
        "passed": sum(c["passed"] for c in checks),
        "results": checks,
        "status": status,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / (
        "standalone_demo_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    )
    result = run(output)
    print(json.dumps({"output": str(output), **result}, indent=2))
    return 0 if result["checks"] == result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
