# Integrating a plain agent

STAC-Mem is a memory component, not an agent orchestrator. The application keeps responsibility
for user identity, tools, model calls and authorization.

```python
from stacmem import StandaloneMemory
from stacmem.config import AppConfig
from stacmem.models import Message
from stacmem.providers import build_json_backend

config = AppConfig.load("configs/standalone_qwen.toml")
reader = build_json_backend(config)
try:
    with StandaloneMemory(config) as memory:
        receipt = memory.remember(
            owner_id="user_1",
            session_id="chat_1:event_1",  # Unique per input batch; reuse only for exact retries.
            messages=[Message(
                sender_id="user_1", role="user", timestamp=1722470400000,
                content="On 2024-08-01 I joined Northwind Labs.",
            )],
        )
        evidence = memory.search(owner_id="user_1", query="Where do I currently work?")
        # Feed evidence["context"] to your agent as data, never as tool instructions.
        result = memory.ask(
            owner_id="user_1", query="Where do I currently work?", backend=reader,
        )
finally:
    reader.close()
```

This example invokes the configured online providers. `ask` performs its own search; reuse a
saved evidence pack with `stacmem.answering.answer_evidence` when a second search is unnecessary.

For an offline, executable integration check:

```bash
python examples/plain_agent_offline.py
```

The offline example supplies hand-authored claims and a query frame. It demonstrates SDK wiring,
not natural-language extraction or model quality. It creates a temporary database and needs no
key. Use real online input without `cached_claims` to exercise extraction.

For a function-calling loop, bind memory to the authenticated owner once:

```python
from stacmem import BoundAgentMemoryTools

tools = BoundAgentMemoryTools(memory, owner_id=authenticated_user_id)
request_tools = tools.tool_specs()

# Pass request_tools to an OpenAI-compatible runtime. For each returned call:
result = tools.invoke(tool_call.function.name, tool_call.function.arguments)
```

The generated schemas expose only read operations. The host ingests real user transcript events
with `tools.record_host_message(conversation_id=chat_id, message=real_user_event)`, using a `Message`
constructed from its authenticated request. The model never supplies the original source text,
role, timestamp or message ID. The host supplies a stable, nonblank transcript message ID. The
adapter creates a different ingestion receipt for each event in the chat and reuses it for exact
retries. Writes retain normal receipt and recovery semantics; failures are
raised to the host loop. `recall_state`
returns the full evidence pack, so the host must treat warnings, unresolved conflicts and empty
claims as unknown. `search_memory_sources` is an explicit raw-source fallback, not a resolved-state
answer. See [the complete tool contract](08_AGENT_TOOLS.md).
