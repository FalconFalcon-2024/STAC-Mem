# Predicate schema

STAC-Mem grants state-changing semantics only to predicates in a validated application registry.
The built-in registry contains `residence.current`, `employment.organization`,
`employment.position`, and the place-scoped `preference.commute_mode`. Applications can add
their own state slots before creating a database.

```toml
[[predicate_schema.predicates]]
name = "contact.channel"
description = "The person's preferred contact channel."
aliases = ["preferred.channel"]
scope = "global"
```

The same immutable registry is used by extraction prompts, query prompts, alias normalization,
source admission, conflict detection, and state resolution. Model output cannot change a slot's
cardinality or scope. Unknown predicates are preserved as quarantined claims but cannot supersede,
correct, retract, or contradict active state. An unsupported state query returns an explicit
`unsupported_predicate` route; applications may separately call `source_search` to inspect original
messages, but source retrieval is not presented as a resolved answer.

Both `StacMemory.from_app_config(config)` and `StandaloneMemory(config)` install this registry and
bind its manifest to the claim ledger. The place registry is bound at the same time, so neither
public entry point can silently reinterpret stored state. `StandaloneMemory` additionally supplies
durable source receipts and atomic prepared recovery and remains the recommended application entry
point for conversation transcripts.

## Supported semantics

Each extension currently has these fixed properties:

- a text value;
- one active value per subject and scope;
- either global scope or a place scope;
- the existing assertion, transition, correction, retraction, temporal, provenance, and
  admission contracts.

Place-scoped slots prefer a trusted `place_id`, then a trusted hierarchy, and finally a normalized
literal name. The model contract accepts only source-grounded names and does not invent canonical
IDs. Applications can configure a trusted alias registry to map grounded multilingual names to a
single identity; unknown names remain literal scopes.

The registry deliberately rejects multi-valued sets, numbers, aggregation, events, and arbitrary
conditions because those require different update and query algebra. Representing them as ordinary
single-value slots would make destructive updates look valid when their semantics are undefined.

Names must be lowercase dotted identifiers. Names and aliases cannot collide with built-in names,
built-in aliases, or another extension. Descriptions tell the model what a slot means; they do not
weaken evidence or factuality checks.

## Database compatibility

The normalized predicate and place registries are stored in the ledger. A database opens only under
compatible registries. Changing a name, alias, description, or scope requires a new database or an
explicit migration; STAC-Mem does not silently reinterpret stored claims. Older standalone ledgers
with matching saved source schemas can establish the new shared ledger binding on first open.
Reordering place aliases does not change the binding.
An empty engine ledger with matching saved predicate/place contracts can bootstrap to
`StandaloneMemory`. A populated direct engine ledger has no standalone source receipts and requires
explicit migration; merely sharing schema manifests does not create missing provenance.

Prepared write batches also contain a policy fingerprint. Recovery therefore requires the same
schema and code policy that prepared the batch. For a custom schema, pass the original config to
`recover-prepared`; the default `--offline` profile intentionally cannot recover it.

See [`configs/custom_schema.toml`](../configs/custom_schema.toml) for a complete provider config.
