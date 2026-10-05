# Agent Tool Integration

`BoundAgentMemoryTools` is the framework-neutral adapter between STAC-Mem and a function-calling
agent loop. It emits read-only OpenAI-compatible function schemas and validates every call before dispatch.
No orchestration framework is required.

## Security boundary

The host application authenticates a user and binds that identity once:

```python
from stacmem import BoundAgentMemoryTools

memory_tools = BoundAgentMemoryTools(memory, owner_id=authenticated_user_id)
```

The model cannot supply or replace `owner_id`. It cannot write source messages or submit
`cached_claims`. The host owns authentication, database access and process isolation.

On receipt of a real user message, the host may write the unchanged event before invoking the
answer model:

```python
from stacmem.models import Message

event = Message(
    sender_id=authenticated_user_id,
    role="user",
    timestamp=server_received_at_ms,
    message_id=transcript_event_id,
    content=actual_user_message,
)
receipt = memory_tools.record_host_message(conversation_id=conversation_id, message=event)
```

The host supplies these values from its authenticated transcript, not from a model completion.
The method requires a nonblank stable `message_id` and rejects a mismatched sender or non-user role.
Each `(conversation_id, message_id)` pair produces a deterministic ingestion receipt ID. A second
message in the same chat gets a new receipt; an identical retry replays the original receipt. Reusing
an event ID with changed content raises `ReceiptConflict`. Use the returned `session_id` for receipt
inspection and recovery.

The advanced `session_id=` argument is retained for explicit per-event ingestion IDs. Reuse it only
for exact retries; do not pass a chat/conversation ID there or combine it with `conversation_id`.
STAC-Mem trusts the host's user identity;
it cannot independently authenticate a fabricated event passed by a compromised host.

## Tools

- `recall_state` performs query compilation, retrieval and deterministic state resolution.
- `search_memory_sources` searches original messages explicitly. It does not claim that a source is
  current or true.

```python
tool_specs = memory_tools.tool_specs()
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tool_specs,
)

for tool_call in response.choices[0].message.tool_calls or []:
    result = memory_tools.invoke(
        tool_call.function.name,
        tool_call.function.arguments,
    )
    # Serialize result as the tool response expected by your agent runtime.
```

Do not hide `IncompleteWrite`, `ReceiptConflict`, validation errors, unresolved-conflict warnings or
an empty claim list. Those signals are part of the memory contract and should cause retry,
clarification or abstention in the host agent.
Claims extracted solely from assistant or tool messages remain in the audit ledger but are
quarantined; these sources do not establish authoritative user state.
The source compiler restores the full referenced user message. Model-selected spans and semantic
hints cannot hide question, modal, negation, fiction or third-party context. Negative hints may veto
a candidate; positive hints do not authorize it. The supported language rules and their limits are
documented in [the grounding contract](09_GROUNDING_CONTRACT.md).

## Place registry

Location-scoped applications should declare stable identities in TOML:

```toml
[[place_schema.places]]
place_id = "ch/be/bern"
name = "Bern"
aliases = ["Berne", "伯尔尼"]
hierarchy = ["CH", "BE", "Bern"]
```

The extractor still has to ground `Bern`, `Berne`, or `伯尔尼` in the source. The trusted registry
then adds the canonical identity. Unknown names remain literal scopes. Changing the registry after
database initialization is rejected because it could reinterpret prior conflict boundaries.

## Context budget

Evidence diagnostics expose:

- `context_budget_method = multilingual-estimate-v1`;
- `context_token_budget`;
- `context_estimated_tokens`;
- `context_truncated`.

This stable offline estimate treats non-ASCII characters conservatively and chunks ASCII words. It
is intentionally not described as an exact model tokenizer. Applications requiring exact billing
or provider limits should apply their provider tokenizer to the returned context as a final check.
