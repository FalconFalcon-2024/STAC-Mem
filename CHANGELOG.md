# Changelog

## Unreleased

- Cross-proposition temporal inheritance requires a positive full-body grammar proof, not just
  the absence of a detected temporal veto; certificates expose accepted/rejected inheritance.
- Layered temporal-evidence accounting detects bare years, period forms and relative durations
  independently of date parsing, and records each occurrence as parsed, object-literal or unresolved.
- Quarter/season/year-part and English/Chinese duration signals cannot silently inherit precise
  event dates; unresolved modifiers still fail conservatively without inferring new intervals.
- Unconsumed date tokens and independent temporal modifiers veto local or inherited date binding;
  certificates retain the blocking source spans and invalid calendar dates stay unsupported.
- Sequential historical ellipsis (`and then at/in`) binds local dates and inherits past-only
  admission rules without upgrading undated historical facts to current state.
- Explicit temporal-scope certificates let leading date adjuncts cover connected change events,
  while local dates, contrast, sentence boundaries and relative-time shifts stop propagation.
- An independent upper-bound guard prevents a new transition value from acquiring another
  state's endpoint even if a proposition connector is not recognized.
- Sequential and subordinate event boundaries plus dated historical ellipsis are supported;
  undated elliptical past states retain the existing conservative admission behavior.
- Undated present states preserve unknown onset while using a separate authenticated assertion-time
  evidence floor for retrieval scoring, point/interval resolution and public conflict comparison.
- Temporal grounding enumerates date occurrences and binds them to the supporting proposition;
  independent action clauses no longer contaminate transition admission for another fact.
- Dated historical state statements support day observations without inferred onset or termination.
- Historical audit reports include evidence-eligibility changes even when validity bounds stay null.
- Explicit OpenAI-compatible chat and embedding endpoints with separate key environments.
- Configured network timeouts and retries, JSON mode control, embedding shape/index validation.
- Read-only session inspection through the CLI and SDK; no replay or repair on inspection.
- Text-only HTTP schema, SDK export and an executable plain-agent example.
- Python 3.11 compatibility target and a 3.11/3.12 CI matrix.
- Explicit replay question time for the strict query compiler.
- Durable prepared batches and an atomic commit for claims, old-version mutations, relations,
  claim FTS and the source-session receipt.
- `recover-prepared` CLI/SDK with integrity/policy checks, no model calls and idempotent replay.
- Crash-injection, rollback and concurrent-recovery regression tests.
- A validated application predicate registry shared by model prompts, canonicalization, admission,
  conflict handling and resolution, with database and prepared-batch compatibility checks.
- Explicit `unsupported_predicate` query routing and conservative quarantine for unknown relations.
- Database-bound embedding identity across provider, model, endpoint, dimensions and explicit
  `space_id`; populated unbound or incompatible ledgers now fail closed.
- Strict configuration models reject unknown fields and revalidate assignments/environment values.
- Prepared journals default to atomic deletion on successful commit, with opt-in forever retention.
- Stable semantic recovery versions replace source-file hashes, and both public factories now apply
  the same predicate schema by default.
- Inactive legacy aliases no longer reserve application predicate names, while release validation
  retains path allowlists and secret checks.
- Iteration-numbered runtime modules and diagnostics now use stable contract/state names. Legacy
  profile labels fail read-only and require an explicit migration, preventing half-migrated state.
- Prepared payloads carry an embedding-space fingerprint, and a durable prepared batch blocks
  vector-space rebinding even before the first claim is committed.
- Recovery-only handles reject writes, embedding-backed state search, answering and empty retries.
- Public state semantics are fixed and validated instead of silently overriding configuration.
- Text-only SDK messages, trusted multilingual place identities, visible multilingual context
  budgeting, and owner-bound function tools tighten application integration boundaries.
- Agent function calls are read-only; authenticated hosts record user transcript events. Claims
  sourced from assistant/tool messages or another sender are quarantined.
- Both public memory entry points bind predicate and place schemas to the claim ledger. Existing
  standalone ledgers can establish this shared binding from their saved source contracts.
- Place alias order no longer changes the normalized registry identity.
- Claim grounding restores the full authenticated user message; model-selected evidence spans
  cannot remove framing context. Semantic hints only restrict admission and transition authority.
- Deterministic question, subject, modality, negation and fiction checks quarantine unsupported
  propositions. Ungrounded transition labels are downgraded; destructive update labels need cues.
- Host transcript tools derive per-event receipt IDs from conversation and message IDs, requiring
  stable message IDs while preserving explicit ingestion-ID compatibility.
- Empty compatible engine ledgers can bootstrap to the source SDK; populated direct ledgers and
  legacy table layouts still require explicit migration. Grounding recovery semantics are versioned.
- `audit-history` evaluates saved proposals against full original transcripts offline.
  `rebuild-history` reconstructs claims and relations into a new database, preserving a source
  snapshot and status/time differences. Intact old prepared proposals can be included explicitly.
- History rebuilds verify receipt/source/batch integrity and both saved schemas. Missing proposals
  or orphaned data block completion; incomplete rebuilds are blocked at both public entry points.
- Source-grounded temporal certificates now clear unsupported model-proposed starts and ends,
  including ambiguous dates and dates attached to a contrasting clause. Explicit dated changes
  still establish their supported onset or termination.
- Current-state admission distinguishes prior and new transition values and quarantines an
  undated past-only state instead of treating it as current.
- Historical rebuilds publish the requested output directory atomically from a sibling staging
  directory. Failed staging ledgers are blocked even when providers fail during initialization;
  audit reports now include conflict-relation graph differences.

Failed sessions without a prepared batch or claims can still use explicit `retry-empty`.
Legacy partially committed sessions and interrupted pre-preparation processing are not
automatically repaired. Recovery rejects incompatible semantic-policy/schema fingerprints.
