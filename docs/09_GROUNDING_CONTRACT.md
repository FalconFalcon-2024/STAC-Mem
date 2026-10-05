# Grounding and State Authority

The application authenticates user messages. The extractor proposes claims, while source contracts
and deterministic language rules decide whether those claims can affect state.

## Original context

The extractor references a real message ID and quotes an `evidence_span`. It may also return
`source_content` as a proposed quote. The compiler retrieves the **full original user message** and
stores that message as the draft's `source_content`. A quote matching only part of the message does
not define the authoritative context.

For example, a quote of `I work at Aster.` from `In a hypothetical example, I work at Aster.`
still produces a nonfactual certificate. A quote of `I work at Aster` from `Do I work at Aster?`
retains the question context and is quarantined.

The source contract records the selected original message ID, value/evidence codepoint offsets,
the proposed model quote, and validation reasons. Offsets are Unicode codepoint positions in the
original message, not byte or tokenizer offsets.

## Conservative semantic hints

| Input | Authority |
|---|---|
| `subject_alignment=aligned` | Cannot override deterministic mismatch or unknown alignment |
| `factuality=asserted` | Cannot override question, modal, negation, or fiction checks |
| `transition_entailment=true` | Cannot create transition evidence |
| Negative or unknown semantic hints | May veto admission or transition |
| Proposed `update_kind=transition` | Downgraded to assertion when change evidence is missing |
| Proposed correction or retraction | Needs independent correction or negative-source cues |

The temporal certificate controls both `valid_start` and `valid_end`. When its source evidence is
insufficient, ambiguous or unsupported, both proposed bounds become unknown; the original model
bounds remain in audit metadata. Supported `since/from`, `before/until`, closed intervals, and an
`On DATE` tied to an explicit departure or arrival can set state-transition bounds.

Dates and transition roles are bound to the value's original proposition, not the whole sentence.
All date occurrences are enumerated before selecting local evidence. Commas, contrast markers,
repeated subjects and coordinated date clauses separate propositions. A date-only introductory
clause may bind to its following proposition. An introductory date may also scope over connected
change events; a date belonging to an independent fact cannot. Multiple local cues or repeated
value mentions do not authorize an arbitrary first-match date.

For example, `Since 2024-08-01 I lived in Paris and on 2025-01-01 I moved to London.` binds London
to the January date. `I lived in Paris until 2025-01-01 and then moved to London.` does not make the
January upper bound London's end: without a locally established onset, London's onset stays
unknown. `I moved to London, and I work at Aster.` does not turn Aster into a relocation destination.
Factuality checks still retain outer questions, hypothetical frames, negation and quotations; action
and date binding uses the narrower proposition.

## Coordinated temporal scope

The temporal certificate records `binding_scope` and `binding_scope_codepoint_span` alongside the
selected cue and value offsets. The supported scopes are:

| Scope | Interpretation |
|---|---|
| `local` | The cue occurs in the value's proposition |
| `introductory` | A standalone date introduction binds the following proposition |
| `coordinated_events` | A leading date adjunct covers a connected change-event group |
| `unresolved` | No supported association; proposed model bounds are cleared |

For example, `On 2025-01-01 I joined Aster and moved to London.` binds the date to both explicit
events, even when the message was recorded in February. The scope is based on literal connectors
and event predicates, not whichever date is closest. An `On DATE` adjunct can cover an explicitly
connected event sequence. A `Since DATE` lower bound is not propagated into a later `then` event.
Date-only introductions and the existing departure/arrival construction remain supported.

Local dates take priority and end the search for an inherited adjunct. Contrast, sentence or
semicolon boundaries, `before/after` subordinate events, and independent relative-time shifts
stop propagation. Postfix dates on another event are not assumed to be a shared introduction.
Historical ellipsis such as `I worked at Aster on DATE1 and at Northwind on DATE2` binds each
observation to its own day. Sequential ellipsis (`and then at Northwind on DATE2`) retains the same
local-date and inherited-past-tense rules; without dates, it cannot acquire current-state authority.

### Unparsed temporal evidence veto

Cue interpretation and temporal-signal detection are separate checks. Before binding a local cue or
inheriting an upstream date, the grounder scans the target proposition for date-shaped tokens and
independent temporal modifiers. The scan also covers intermediate propositions and the proposed
shared-date anchor. A token is consumed only when its original offsets fall within a supported cue
or the literal claim value; matching the same calendar value is not enough.

An unconsumed date or temporal modifier blocks the binding, records `binding_scope=unresolved`, and
clears both model-proposed validity bounds. For example:

```text
On 2025-01-01 I joined Aster and moved to London by 2025-03-01.
On 2025-01-01 I joined Aster and by 2025-03-01 moved to London.
On 2025-01-01 I joined Aster and eventually moved to London.
On 2025-01-01 I joined Aster and moved to London the following month.
```

None authorizes London's onset to be January 1. Recognizing `by DATE`, a relative shift or a named
month as a signal does **not** implement its interval semantics. The onset remains unknown instead
of assigning an unsupported date. The veto also applies when an unrecognized connector leaves the
leading date and target in the same proposition, or when a date-only introduction precedes the target.

Certificates retain `unresolved_temporal_signal_spans` and
`unresolved_temporal_signal_codepoint_spans` with the reason
`independent_temporal_signal_blocks_binding`. Invalid calendar dates remain unparsed evidence in
the v2 contract rather than becoming authoritative bounds. Supported cues consume their own date
tokens, including both endpoints of a supported interval. Tokens within the literal object value
do not by themselves create a veto, for example a company named `Eventually Labs`.

The signal scanner covers common numeric/ISO, slash-separated and named-month forms plus a finite
English/Chinese modifier vocabulary. Evidence accounting additionally checks standalone four-digit
year tokens, period forms and compositional duration/direction expressions independently of that
vocabulary. For example, an unfamiliar qualifier in `the glorp phase of 2037` cannot hide the year
token and authorize an inherited onset. Bare four-digit tokens can also represent amounts; when
they are outside a supported cue or literal value, the conservative policy leaves time unknown
rather than guessing which interpretation applies.

The evidence layers are:

| Kind | Examples; detection does not assert interval meaning |
|---|---|
| `date_form` | ISO/slash dates, named months, Chinese numeric dates |
| `bare_year` | A standalone four-digit token, including a year adjacent to Chinese text |
| `relative_duration` | Unit plus relative direction: `two months afterward`, `几个月后` |
| `period` | `Q1`, seasons, halves/parts of a year, `年中`, `第一季度`, `下半年` |
| `modifier` | Independent shifts and relative periods such as `eventually`, `next semester` |

The v2 certificate includes `temporal_evidence_detector=temporal-evidence-accounting-v1` and
`temporal_evidence_accounting`. Every detected evidence occurrence in the checked propositions
has its original span, codepoint range, kind and disposition: `parsed_cue`, `object_literal`, or
`unresolved`. Full literal containment is required; a partially covered expression or the same
year elsewhere cannot be treated as consumed. An `unresolved` item vetoes binding. The existing
unresolved-signal fields remain available for callers that only need blocking evidence.

`in Q1 2025`, `in spring 2025`, `during the summer of 2025`, `toward the end of 2025`,
`two months afterward`, `两个月后` and `年底` therefore do not acquire January 1 onset from a
previous event. This does not infer the start of the quarter/season or calculate a relative date.
`after lunch` and `after telling my family` deliberately remain unresolved: recovering that
recall requires a separately validated compatibility rule, not simply removing the veto.

This is still a conservative, bounded temporal-language recognizer, not an exhaustive parser. It
cannot prove that every arbitrary natural-language time expression will be detected and may
reject a binding in complex clauses rather than prove that multiple expressions are compatible.
When time is rejected but the factual transition remains grounded, the claim may still be admitted
with unknown onset and the authenticated assertion-time evidence floor described below. This is not
a guarantee of an exact date, and it does not permit retrospective answers before that evidence floor.

`I left Northwind on DATE and joined Aster` still leaves Aster's onset unknown: a postfix departure
date is not assumed to date the arrival.

An additional invariant is independent of the proposition splitter: a `new` transition value
cannot receive a pure upper bound without direct endpoint evidence for that value's state.
`I lived in Paris until DATE then moved to London` and an unrecognized connector such as
`... until DATE upon moving to London` cannot make London a state that ends before its arrival.
Likewise, `I joined Aster before DATE` places an upper bound on an arrival event, not on the end of
employment; it does not authorize `valid_end=DATE`. Unsupported dates stay in the certificate's
audit fields, while authoritative bounds are cleared. Independent historical endpoint statements
such as `I lived in London until DATE` remain supported.

These are conservative temporal-scope rules for a supported grammar, not a full natural-language
dependency parser. Complex coordination or indirect references can still leave time unknown.

## Unknown onset and dated observations

An undated present assertion such as `I work at Aster.` does not establish a real-world onset.
Its `valid_start` stays `null`. `metadata.temporal_eligibility` records an **evidence eligibility
floor** derived from the authenticated message's assertion time, not from a model-proposed date.
Resolution excludes it before that time and permits continuing-state evidence from that time
onward, subject to scope, conflicts and subsequent versions. This is a state-continuity convention,
not proof of an uninterrupted real-world state or of a start event.

Retrieval scoring, point/interval projection and public conflict comparison use the same evidence
interval. The answer context renders `valid=unknown` and `evidence_from=... (not an onset)`.
An explicit source-grounded historical upper bound remains a historical bound; it is not replaced
by the later message time. Knowledge time independently controls when the system knew a version.

`On 2024-08-01 I worked at Aster.` and `As of 2024-08-01, I still work at Aster.` support a
day-grain `holds_at` observation, represented as `[2024-08-01, 2024-08-02)`. Observation metadata
explicitly says `asserts_onset=false` and `asserts_termination=false`. Those day boundaries limit
the evidence, not employment onset or termination. Such an observation does not end another
supported continuing state and does not by itself justify a future current-state answer.

For directional updates, the certificate also identifies whether the proposed value is `prior`
or `new`. A claim that proposes the prior value as a current state is quarantined. `I worked at
Aster.` has no established end date and is likewise not eligible for current-state resolution;
an explicit bounded historical interval remains usable as history.

`My employer is Aster.` is an assertion. If an overlapping prior employer differs, the system
records a contradiction rather than selecting the newer value solely because the model labeled it
a transition. `I left Northwind and joined Aster.` contains directional change evidence and can
supersede a supported prior state under the time and scope rules.

## Auditable decisions

The proposition certificate records the code-selected support clause and offsets, source context,
object grounding, subject alignment, factuality reasons, value direction, transition reasons, and whether validated
model hints were considered. It never uses the model's shortened evidence as a replacement for the
source context. Temporal grounding likewise locates the original source clause rather than trusting
model-chosen boundaries.

Quarantine preserves the original message and candidate. It creates no conflict edges or mutations
of prior state. The host can inspect the reasons or retrieve original sources explicitly.

## Supported language boundary

The rules support common English and Chinese self statements, explicit named subjects, questions,
modal and negative statements, hypothetical/fictional framing, and directional change cues. They
are a conservative, inspectable grammar; they do not prove arbitrary natural-language entailment
or authenticate the truth of a user's statement. Indirect discourse, complex coreference or an
unknown subject may be quarantined. Unknown state predicates also remain quarantined until the
application declares their semantics.

The advanced direct claim engine trusts the host-supplied drafts and their provenance contracts.
Applications accepting user conversations should use `StandaloneMemory` and the host-only
`BoundAgentMemoryTools.record_host_message(conversation_id=..., message=...)` entry. A stable
transcript `message_id` is required. The model receives only read tools.

## Upgrading a database

New writes use the full-message grounding policy. Existing committed claims are not retroactively
reclassified by normal startup. Use `audit-history` to evaluate historical proposals under the
current policy and `rebuild-history` to create a new database with reconstructed state relations.
Both read authentic original messages; neither modifies the original ledger.

Prepared batches include a semantic-policy fingerprint. `recover-prepared` preserves the original
transaction policy and rejects a different fingerprint. The explicit `rebuild-history
--include-prepared` path instead imports intact old proposals, revalidates them, rebuilds vectors,
and creates new receipts under the current policy. It does not require the old runtime for the
supported saved source/batch format. See [history upgrades](10_HISTORY_UPGRADE.md).
