"""Primary command-line interface for STAC-Mem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig
from .models import Message, QueryFrame
from .standalone import StandaloneMemory


def main() -> int:
    parser = argparse.ArgumentParser(description="STAC-Mem state memory")
    provider = parser.add_mutually_exclusive_group(required=True)
    provider.add_argument(
        "--offline", action="store_true", help="No network; cached structure only"
    )
    provider.add_argument("--config", type=Path, help="Online provider config (may incur API cost)")
    parser.add_argument("--database", type=Path, required=True, help="STAC-Mem database path")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    inspect = commands.add_parser("inspect", help="Read-only session diagnosis; no provider calls")
    inspect.add_argument("--owner", required=True)
    inspect.add_argument("--session", required=True)
    recover = commands.add_parser(
        "recover-prepared", help="Commit saved drafts/vectors without model calls"
    )
    recover.add_argument("--owner", required=True)
    recover.add_argument("--session", required=True)
    audit = commands.add_parser("audit-history", help="Revalidate historical proposals offline")
    audit.add_argument("--include-prepared", action="store_true")
    audit.add_argument("--report", type=Path, help="Save a new audit report without overwriting")
    rebuild = commands.add_parser("rebuild-history", help="Revalidate into a separate new database")
    rebuild.add_argument("--output", type=Path, required=True)
    rebuild.add_argument("--include-prepared", action="store_true")
    rebuild.add_argument("--allow-paid-api", action="store_true")
    retry = commands.add_parser("retry-empty", help="Explicitly retry a failed zero-claim session")
    retry.add_argument("--owner", required=True)
    retry.add_argument("--session", required=True)
    retry.add_argument("--allow-paid-api", action="store_true")
    serve = commands.add_parser("serve", help="Local development API on 127.0.0.1 only")
    serve.add_argument("--port", type=int, default=8020)
    serve.add_argument("--allow-cached-claims", action="store_true")
    remember = commands.add_parser("remember")
    remember.add_argument("input", type=Path)
    remember.add_argument("--allow-cached-claims", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("input", type=Path, help="owner_id, query, optional frame, top_k")
    ask = commands.add_parser("ask", help="User-facing state answer with evidence")
    ask.add_argument("input", type=Path)
    ask.add_argument("--allow-paid-api", action="store_true")
    source = commands.add_parser("sources")
    source.add_argument("--owner", required=True)
    source.add_argument("--query", required=True)
    args = parser.parse_args()
    if args.command in {"audit-history", "rebuild-history"}:
        from .migration import MigrationBlocked, revalidate_history

        config = AppConfig.load(args.config) if args.config else None
        try:
            result = revalidate_history(
                args.database, config=config,
                output=args.output if args.command == "rebuild-history" else None,
                include_prepared=args.include_prepared,
                allow_paid_api=getattr(args, "allow_paid_api", False),
            )
        except MigrationBlocked as exc:
            print(json.dumps(exc.report, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "audit-history" and args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        return 0 if result["status"] == "completed" else 2
    if args.command == "inspect":
        from .inspection import inspect_session

        result = inspect_session(args.database, owner_id=args.owner, session_id=args.session)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "ask" and (args.offline or not args.allow_paid_api):
        parser.error("ask requires --config and --allow-paid-api; use demo for offline checks")
    if args.command == "retry-empty" and not args.offline and not args.allow_paid_api:
        parser.error("Retrying online extraction may incur API cost; pass --allow-paid-api")
    if args.command == "recover-prepared":
        # Recovery uses persisted vectors and keeps only the configured state policy.
        config = AppConfig.load(args.config) if args.config else AppConfig()
        config.runtime.database_path = str(args.database)
        config.extraction.provider = "rule"
        config.embedding.provider = "hash"
        config.rerank.provider = "none"
        memory = StandaloneMemory.recovery(config)
    elif args.offline:
        memory = StandaloneMemory.offline(args.database)
    else:
        config = AppConfig.load(args.config)
        config.runtime.database_path = str(args.database)
        memory = StandaloneMemory(config)
    with memory:
        if args.command == "serve":
            import uvicorn

            from .standalone_api import create_app

            if not 1 <= args.port <= 65535:
                parser.error("port must be between 1 and 65535")
            uvicorn.run(
                create_app(memory, allow_cached_claims=args.allow_cached_claims),
                host="127.0.0.1",
                port=args.port,
            )
            return 0
        if args.command == "status":
            result = memory.status()
        elif args.command == "recover-prepared":
            result = memory.recover_prepared(owner_id=args.owner, session_id=args.session)
        elif args.command == "retry-empty":
            result = memory.retry_failed_empty(owner_id=args.owner, session_id=args.session)
        elif args.command == "sources":
            result = memory.source_search(owner_id=args.owner, query=args.query)
        else:
            data = json.loads(args.input.read_text(encoding="utf-8-sig"))
            if args.command == "remember":
                if "cached_claims" in data and not args.allow_cached_claims:
                    parser.error(
                        "Cached claims require --allow-cached-claims; not model extraction"
                    )
                result = memory.remember(
                    owner_id=data["owner_id"],
                    session_id=data["session_id"],
                    messages=[Message.model_validate(m) for m in data["messages"]],
                    cached_claims=data.get("cached_claims"),
                )
            elif args.command == "ask":
                from .providers import build_json_backend

                backend = build_json_backend(config)
                try:
                    result = memory.ask(
                        owner_id=data["owner_id"],
                        query=data["query"],
                        backend=backend,
                        frame=QueryFrame.model_validate(data["frame"])
                        if data.get("frame")
                        else None,
                        top_k=data.get("top_k", 10),
                    )
                finally:
                    backend.close()
            else:
                result = memory.search(
                    owner_id=data["owner_id"],
                    query=data["query"],
                    frame=QueryFrame.model_validate(data["frame"]) if data.get("frame") else None,
                    top_k=data.get("top_k", 10),
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
