# Context: lang-graph-state

## Purpose

A LangGraph flow that fetches data from a fixed set of upstream APIs in parallel,
then uses an LLM to synthesise the results into a response.

---

## Glossary

### Graph

The LangGraph `StateGraph` that orchestrates the full fetch → synthesise pipeline.

### Static Source

An upstream API that is always queried on every run, regardless of the input.
The set of static sources is fixed at design time.
Current static sources: **Customer Profile**, **Insights**, **Optimizer**, **Monte Carlo**.

### Synthesis

The LLM step that takes the aggregated results from all static sources and produces
the final output. Runs after all sources have responded.

### TokenManager

A per-API component responsible for obtaining and caching a bearer token from the
shared OAuth token server. Tokens are cached for a fixed TTL (15 minutes).
Each static source has its own `TokenManager` backed by its own credentials
(username/password from environment variables).

### HTTP Client

An `httpx.AsyncClient` instance, one per static source, injected into each node
via closure at graph construction time. The client is constructed once at startup
and shared across invocations of that node.

### Node Factory

A function that accepts an HTTP client and a `TokenManager` and returns an async
LangGraph node function. Enables isolated unit testing by accepting fakes at
construction time.

### Graph State

A flat Pydantic model with one typed field per static source plus the synthesis
output. Each fetch node writes exactly one field. The synthesis node reads all
source fields and writes the final output field.

### Source Model

A Pydantic model representing the parsed, validated result of one static source's
API response. Each static source has its own `SourceModel`. Parsing happens inside
the fetch node — raw HTTP responses never appear in state.

### Synthesis Output

A structured Pydantic model produced by the LLM synthesis node. Returned on success.

### LLM Gateway

A REST API used for synthesis. Called with a JSON body of shape:
`{ model: { provider, id }, prompt_spec: { messages: [{ role, content }] } }`.
Accessed via the same `httpx.AsyncClient` + `TokenManager` pattern as the static
sources. The model `provider` and `id` are supplied via environment variables.

### CheckpointerFactory

A function `make_checkpointer()` that reads an environment variable to decide which
checkpointer backend to return — `MemorySaver` for local/test, Postgres for
production. No graph or node code references a specific checkpointer implementation
directly.

### fs_req_id

The caller-supplied identifier for a graph invocation. Passed as the LangGraph
`thread_id` in `config["configurable"]["thread_id"]`. Scopes the checkpoint to a
single request. Correlates graph runs with upstream request tracking.

### SourceFetchError

A typed exception raised by a fetch node when its upstream API call fails.
Carries `source: str` (the source name) and `cause: Exception`.
The graph aborts and propagates this exception to the caller — synthesis is skipped.
The caller is responsible for catching `SourceFetchError` and handling it.
