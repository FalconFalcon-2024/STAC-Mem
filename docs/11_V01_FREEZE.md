# v0.1 Contract Freeze

The release freezes correctness boundaries, not a promise to parse arbitrary language.

## Supported Temporal Structures

`DATE` means a valid, explicit `YYYY-MM-DD` source literal. Examples describe the supported
canonical structures; they do not grant authority to arbitrary surrounding prose.

| Structure | Meaning |
| --- | --- |
| `Since DATE I work at Aster.` | Source-grounded lower bound |
| `From DATE1 to DATE2 I worked at Aster.` | Explicit bounded historical interval |
| `As of DATE, I still work at Aster.` | Day observation, not an onset |
| `On DATE I worked at Aster.` | Day observation, not a permanent state |
| `On DATE I moved from Paris to London.` | New-value onset, prior-value role checked separately |
| `On DATE I left Northwind and joined Aster.` | Supported connected departure/arrival scope |
| `On DATE I joined Aster and moved to London.` | Supported connected event scope |

Local date association retains independent temporal-evidence accounting and endpoint checks.
**Cross-proposition inheritance** additionally requires a full-body grammar proof for the target,
intervening events and dated anchor. It does not depend solely on a finite list of unsafe words.
The inherited target must match its exact quoted object literal. Other event entities must fit
the restricted proper-name shape implemented in `_supported_scope_body`; arbitrary entity tails
and arbitrary modifiers are not accepted. The positive grammar also retains the previously
supported literal Chinese residence and departure/arrival structures, and explicitly scoped
commute statements. It does not infer arbitrary Chinese conjunction or relative-time scope.

This is deterministic recognition within a restricted grammar, not a universal semantic proof.
Unsupported structures, ambiguous relative time and unknown adjuncts may keep both bounds null.
Losing a guessed date is preferable to manufacturing a trusted date. Unsupported time is not
automatically a false statement: admission and temporal support are separate decisions.

An undated accepted present assertion can be eligible from its authenticated assertion-time
evidence floor. That floor is neither a real-world onset nor permission to answer arbitrary past
queries. A past-only unsupported assertion can instead remain quarantined.

Certificates expose `inheritance_proof=complete_body_v1` or `unsupported_body`; local bindings
use `not_required`. Accepted/rejected source spans remain auditable. Positive model hints do not
bypass these checks. New policy identity: `positive-temporal-scope-v8`.

## Frozen Correctness Invariants

| Invariant | Regression evidence |
| --- | --- |
| Models cannot fabricate authoritative original source | `test_source_authority.py`, `test_contracts.py` |
| Positive model hints cannot raise authority | `test_grounding.py`, `test_admission.py` |
| Unsupported inherited scope cannot authorize a precise bound even if the lexical detector misses it | `test_temporal_positive_scope.py` |
| Unknown onset cannot project arbitrary past | `test_temporal_propositions.py` |
| A transition's prior value cannot become its new current state | `test_grounding.py`, `test_temporal_grounding.py` |
| Schema and embedding mismatches fail closed | `test_predicate_schema.py`, `test_operational_boundaries.py` |
| Claims, relations and receipt commit atomically | `test_atomic_sessions.py`, `test_standalone_recovery.py` |
| A failed historical rebuild cannot be opened as a valid rebuilt DB | `test_history_migration.py` |
The tests are correctness/regression evidence, not evidence of universal
language coverage. Existing historical databases are not silently rewritten. Keep the original,
audit it, and explicitly rebuild into a new database when changing policy. Prepared batches retain
their compatibility checks; a fingerprint mismatch is not an invitation to force replay.
