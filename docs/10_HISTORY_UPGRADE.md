# Revalidating Historical Memory

Upgrading code changes future writes. Existing claims and saved prepared batches are durable data;
their old status is not silently rewritten when an application starts.

An explicit history upgrade reuses saved candidate proposals and their authentic original messages,
applies the current admission, time, place and conflict rules, and reconstructs the whole version
chain. If an old hypothetical claim incorrectly superseded a true prior claim, the rebuilt ledger
can quarantine the hypothetical and restore the prior claim's status and open interval.

## Offline audit

Supply the actual existing database path. For a custom predicate or place registry, use the original
`--config` instead of `--offline`; audit still uses local hash embeddings without provider calls.

```bash
python -m stacmem.standalone_cli --offline --database runtime/memory.sqlite3 \
  audit-history --include-prepared --report runs/history-audit.json
```

Audit uses SQLite's backup API to obtain a consistent snapshot, then builds a temporary ledger. It
prints final status/time and evidence-eligibility changes, relation graph
additions/removals/type changes, quarantine reasons, counts, blocked sessions and the snapshot
hash. `--report` saves a new report and refuses to overwrite an existing file. Audit assesses state
policy changes; its local hash vectors are not an evaluation of online retrieval quality.

## Rebuild into a new directory

To keep an offline hash-embedding deployment:

```bash
python -m stacmem.standalone_cli --offline --database runtime/memory.sqlite3 \
  rebuild-history --include-prepared --output runs/history-upgrade
```

To rebuild with the configured online embedding model:

```bash
python -m stacmem.standalone_cli --config configs/standalone_qwen.toml \
  --database runtime/memory.sqlite3 rebuild-history --include-prepared \
  --output runs/history-upgrade --allow-paid-api
```

This path does not call the extraction model again. It rebuilds vectors in the selected embedding
space; online embeddings require explicit cost opt-in. An offline rebuild produces a hash-vector
database that should subsequently be opened with the matching offline profile. The online rebuild
retains the selected embedding configuration for later normal online application use.

The output directory must not exist. Rebuild uses a hidden sibling staging directory on the same
filesystem. The requested output path appears only after the database, snapshot and report are
complete and the staging directory is renamed. On success it contains:

- `memory.sqlite3`: the new source receipts, revalidated claims and reconstructed relations;
- `source_snapshot.sqlite3`: the consistent original database snapshot;
- `report.json`: changes and provenance of the upgrade.

Only switch the application's database path after `report.json` says `status=completed` and the
differences have been reviewed. Stop old writers when switching so later writes to the old database
are not lost. Source snapshots and reports contain user data and belong outside a public release.
The upgrade does not automatically redirect a running application.

## Old prepared batches

`recover-prepared` preserves the exact semantics of an already prepared transaction; its policy
fingerprint must match. `rebuild-history --include-prepared` treats the old batch as saved proposals
to be checked anew. It verifies its checksum, request identity, candidate ownership, source records
and embedding identity, then applies the current contracts to the full messages. New receipts are
created in the new ledger; the old batch remains untouched.

This supports source-linked `standalone-state-v1` ledgers and `atomic-prepared-v2` batches, including
the earlier grounding policy. Unrelated legacy table layouts require a separate format migration.

## Completion checks

The upgrade verifies original request hashes, persisted source messages, committed receipt/claim
links, pending payload hashes, and both saved predicate/place contracts. It refuses an existing
output directory, unlinked claims or orphaned sources. A processing session without a saved batch,
failed extraction with no saved candidates, partial old claims or missing provenance blocks a
complete rebuild. Recover or re-extract those inputs explicitly before rerunning into a new output.

A failure leaves a hidden `.OUTPUT.staging-*` directory with `report.json` marked `failed` and
the actual failure type. Even provider initialization failures mark its ledger unusable. Both
`StandaloneMemory` and the public `StacMemory` factory refuse that partial ledger. The requested
output path remains absent and the original database remains available. Inspect or retain the
staging directory as needed; subsequent attempts use a fresh staging directory.
