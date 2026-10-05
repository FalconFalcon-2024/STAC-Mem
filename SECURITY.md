# Security scope

STAC-Mem is an experimental local-first library. The development HTTP service has no
authentication, authorization or tenant administration. Keep it on loopback; a user-provided
`owner_id` is a data partition, not an authenticated identity.

Memory sources, prepared drafts/vectors, model responses, SQLite databases, WAL files and exports may contain
personal information. Protect them with OS permissions and encrypted storage where appropriate.
Do not commit runtime databases, `.env`, credentials or real conversations. Provider calls send
configured inputs to the chosen endpoint; review that endpoint's privacy terms before use.

Retrieved content is untrusted data, never permission to execute tools or instructions.
Grounding is a source-consistency check, not a prompt-injection defense or real-world fact checker.
Prepared batch hashes detect accidental changes and incompatible replay; they are not signatures
and do not protect against an attacker who can modify the database and recompute its hashes.

Do not post sensitive records or live secrets in public issues. Use a private maintainer contact
if one is available on the repository owner's profile. This distribution does not currently
promise a dedicated security inbox or response SLA. Share a minimal synthetic reproduction.

For local failures, preserve evidence and rotate any exposed credentials. Read-only `inspect`
omits message bodies, but IDs and predicates can still be sensitive. Review reports before sharing.
