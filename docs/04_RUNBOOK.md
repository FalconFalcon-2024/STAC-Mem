# STAC-Mem Runbook

## One-command local demo

Python 3.11 or newer is required. The command creates `.venv`, installs STAC-Mem, and runs a
deterministic lifecycle demonstration.

Linux and macOS:

```bash
bash scripts/quickstart.sh
```

Windows PowerShell:

```powershell
.\scripts\quickstart.ps1
```

Cross-platform alternative:

```bash
python quickstart.py
```

The result is written under `runs/standalone_demo_*` and contains `summary.json`, `records.jsonl`,
and an isolated SQLite ledger.

## Development verification

```bash
python quickstart.py --dev
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests scripts quickstart.py
```

Use `.venv\Scripts\python.exe` on Windows.

## Natural-language CLI

Set `DASHSCOPE_API_KEY` in the process environment, then run:

```bash
.venv/bin/python -m stacmem.standalone_cli \
  --config configs/standalone_qwen.toml \
  --database runtime/stacmem.sqlite3 \
  remember examples/standalone_session.json

.venv/bin/python -m stacmem.standalone_cli \
  --config configs/standalone_qwen.toml \
  --database runtime/stacmem.sqlite3 \
  search examples/standalone_query.json
```

## Local API

```bash
export DASHSCOPE_API_KEY="..."
bash scripts/start_stacmem.sh
```

Or copy `.env.example` to `.env`, fill in `DASHSCOPE_API_KEY`, and start it directly. On Windows:

```powershell
.\scripts\start_stacmem.ps1
```

The service binds to `127.0.0.1:8020`. Open `http://127.0.0.1:8020/docs` for the API schema and
run `bash scripts/healthcheck.sh` for a health check. This development server is loopback-only and
does not claim production authentication or multi-tenant isolation.

## Troubleshooting

- Python version mismatch: install Python 3.11 or newer and rerun the quickstart.
- `DASHSCOPE_API_KEY is required`: the deterministic demo does not need a key; natural-language
  ingestion does.
- Existing output directory: choose a new `--output` path or let the demo generate one.
- Failed or interrupted write: inspect the receipt before using `retry-empty`; do not replay writes
  blindly.

## Inspect a failed session without a key

```bash
python -m stacmem.standalone_cli --offline --database runtime/stacmem.sqlite3 inspect --owner user_1 --session session_1
```

This opens the existing database read-only. It does not load model credentials or replay writes.
The result lists receipt status, claim IDs/statuses, blocking sessions and recovery eligibility.
Raw message text, values, provider exceptions and saved request bodies are not printed.

- `explicit_retry_empty`: the failed session currently has no claims and no other blocking session.
  The existing `retry-empty --owner ... --session ... --allow-paid-api` command rechecks these
  conditions and may repeat a paid extraction. A saved prepared batch prevents this retry.
- `explicit_recover_prepared`: use `recover-prepared` to commit saved drafts and vectors.
  Inspection only identifies a candidate; recovery validates integrity and policy compatibility.
- `preserve_database_partial_recovery_not_supported`: some claims were committed. Preserve the
  database and its WAL while stopping writers; do not delete the receipt or mark it committed.
- `check_writer_liveness_do_not_replay`: a processing receipt may belong to a live writer.
  This command intentionally does not assume it is dead.

Inspection is a snapshot, not a lock or authorization to force recovery. Use the SDK's
`memory.inspect_session(owner_id=..., session_id=...)` for the same report.

## Recover a prepared batch

```bash
python -m stacmem.standalone_cli --config configs/standalone_qwen.toml --database runtime/stacmem.sqlite3 recover-prepared --owner user_1 --session session_1
```

This command does not require provider keys or invoke models: it loads the configured state policy
but uses the persisted vectors and drafts. With the default state policy, `--offline` can replace
`--config ...`. A custom predicate or place schema requires its original config. SDK callers use
`memory.recover_prepared(owner_id=..., session_id=...)`.
The SDK recovery handle permits status, session inspection, literal source search and prepared
recovery only. It rejects new writes, state search, answering and failed-empty retries so its local
hash embedder cannot contaminate a provider-bound ledger.

Predicate and place schemas are bound when the database is created. Do not edit registered names,
aliases, descriptions or scope in place. Use a new database or an explicit migration; see
[`07_PREDICATE_SCHEMA.md`](07_PREDICATE_SCHEMA.md).

Recovery commits the entire batch or none of it, including changes to old versions and the receipt.
Successful identical retries return the saved result. A response lost after commit does not turn
the session back into a failed one. The original writer may still be alive after preparation;
its later commit checks the same receipt and cannot duplicate the recovered batch.

If extraction/embedding was interrupted before preparation, the receipt may still be `processing`.
There is no automatic takeover for that phase. Legacy partial claims, another incomplete session,
checksum mismatch or changed semantic-policy/schema fingerprint stop recovery. Preserve the database
and use the matching software/configuration for diagnosis; do not edit checksums or force status
to committed. Stop old-version writers before updating the software.

Prepared payloads contain source-linked drafts and vectors. The default `until_commit` retention
deletes them in the same transaction that commits claims and the receipt. If the commit fails, the
deletion also rolls back, so recovery remains possible. Set `[recovery]
prepared_retention="forever"` only when an audit/debug deployment accepts the additional storage.
Back incomplete journals up together with the SQLite database using a consistent backup, or stop
all writers before copying the database and its WAL. They are not public release files.

The claim ledger also binds its embedding provider, model, endpoint hash, dimensions and `space_id`.
Changing any of them after the first claim requires a new database or explicit re-embedding
migration. Startup intentionally refuses a legacy populated ledger without that binding.

The previous `standalone-source-v1+contracts-v15` metadata label is not rewritten automatically.
Startup rejects it before schema creation or metadata mutation. Back up the database and use an
explicit migration/export process, or start with a new database.

## Concurrency boundary

One local writer process per database is the recommended deployment. The SDK reserves incomplete
owners and SQLite serializes final commits and recovery attempts across instances. It does not
provide a distributed writer lease, network-filesystem guarantee or automatic recovery of legacy
partial writes. Source acceptance and preparation precede the final atomic state/receipt commit.
Low-level ledger writes bypassing the standalone SDK do not receive its session safeguards.
An empty, schema-bound `StacMemory` ledger may bootstrap to `StandaloneMemory` under matching
registries. Existing direct claims, relations or commits require explicit migration before the
SDK can manage source receipts. Older table layouts are rejected even when empty.

The grounding update changes the semantic recovery fingerprint. A prepared batch from the earlier
grounding policy cannot be directly replayed through `recover-prepared` under this policy.
`rebuild-history --include-prepared` can revalidate supported old candidates from their authentic
original messages into a separate database under the current policy. This also reclassifies
historical claims and reconstructs prior-version mutations. See [the upgrade procedure](10_HISTORY_UPGRADE.md).

The local HTTP API accepts nonempty text messages only. Its schema rejects structured media
content before ingestion. Do not expose the unauthenticated development server to the Internet.
