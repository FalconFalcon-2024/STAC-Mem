# Provider configuration

The model proposes claims or a query frame. Changing a provider does not bypass source grounding,
admission, temporal contracts or state resolution. An API that accepts chat requests is not
necessarily capable of producing reliable structured claims.

## Presets and generic endpoints

`configs/standalone_qwen.toml` keeps the existing Qwen preset. A Qwen key is read from
`DASHSCOPE_API_KEY`; its default URL is the DashScope compatible endpoint.

`configs/openai_compatible.toml` is a template for explicit compatible endpoints, including local
servers. Replace both model names, both URLs and the embedding dimension before use. Set the
environment variables named in `api_key_env`; do not put keys in TOML. For a local service that
ignores authentication, an explicit dummy environment value may be used.

| Setting | Meaning |
| --- | --- |
| `extraction.provider` | `qwen`, `openai_compatible`, or offline `rule` |
| `embedding.provider` | `qwen`, `openai_compatible`, or offline `hash` |
| `base_url` | Independent chat/embedding address; explicit config precedes `STACMEM_BASE_URL` |
| `api_key_env` | Independent credential environment variable for each endpoint |
| `extraction.json_mode` | Send `response_format=json_object`; disable only if unsupported |
| `embedding.send_dimensions` | Omit the dimensions request field for fixed-dimension servers |
| `embedding.dimensions` | Expected vector size; responses are still checked when field is omitted |
| `embedding.space_id` | Stable application label for the exact embedding space |
| `runtime.request_timeout_seconds` | SDK request timeout for extraction, query, answers and embeddings |
| `max_retries` | SDK retries for each provider; independent chat and embedding settings |
| `rerank.provider` | `none` or the existing `dashscope` adapter, with its separate timeout |

A generic provider without an explicit URL fails before any network call rather than selecting
a remote vendor implicitly. A model returning a non-object JSON response is rejected. Embedding
responses are reordered by input index and checked for missing indices, dimensions and finite
values before persistence.

The ledger binds a secret-free embedding identity before its first claim. The identity contains
provider, model, dimensions, dimensions-request behavior, an endpoint hash, and `space_id`. After
claims exist, any mismatch stops startup before retrieval or writes. Equal dimensions therefore do
not make two models or endpoints compatible. A durable prepared batch also counts as
vector-space-dependent state: its payload records the embedding-space fingerprint and prevents
rebinding or cross-space recovery before the first claim. Use a new database or an explicit
re-embedding migration; deleting the binding is not a migration. A ledger with neither claims nor
prepared batches may be reconfigured. Set an explicit `space_id` in deployment configs so
operational intent is reviewable.

Timeout is an SDK network timeout, not a hard deadline for an entire session: retries, multiple
extraction requests and later storage work can make total time longer. Use `max_retries=0` while
diagnosing a connection. Disabling JSON mode does not add a silent text fallback: the returned
content must still parse as a JSON object.

## Custom backends

`stacmem.providers.JsonBackend` describes `model`, `last_usage` and `call(system, user) -> dict`.
It can be used by `ContractClaimExtractor`, `ContractQueryCompiler` or `memory.ask`.
`stacmem.embeddings.Embedder` describes `dimensions` and `embed(texts)`.
These interfaces do not imply support for every vendor's native API. The shipped generic adapter
uses compatible chat completions and embeddings, not native Gemini or other proprietary schemas.

For replayed questions, `ContractQueryCompiler.compile(..., asked_at=timestamp_ms)` binds
"current" to the supplied question time. This is independent of when the program is run.
Relative-date interpretation remains governed by the explicit temporal contract.

All TOML configuration models reject unknown fields, and assignment/environment overrides are
revalidated. A misspelled provider URL or a non-positive dimension fails startup instead of being
silently ignored.
