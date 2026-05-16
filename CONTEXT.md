# Context: lang-graph-state

## Purpose

A LangGraph flow that fetches data from a fixed set of upstream APIs in parallel,
then uses an LLM to synthesise the results into a response.

---

## Glossary

### Graph

The LangGraph `StateGraph` that orchestrates the full fetch → synthesise pipeline.
Acquired via `build_graph(...)` as an **async context manager**. Entering the
context constructs the six owned `HTTP Client`s (see below) and initialises every
`TokenManager`; exiting closes all clients cleanly. Callers must use the context
manager — the compiled graph is not safely usable outside it.

### Static Source

An upstream API that is always queried on every run, regardless of the input.
The set of static sources is fixed at design time.
Current static sources: **Customer Profile**, **Insights**, **Optimizer**, **Monte Carlo**.

### Synthesis

The LLM step that takes the aggregated results from all static sources and produces
the final output. Runs after all sources have responded.

Built with the same pattern as a fetch node: two pure callables —
`build_synthesis_request(state) -> RequestSpec` and
`parse_synthesis_response(response) -> SynthesisOutput` — are injected into
`make_synthesis_node`, which handles the token, header injection, HTTP call,
and error wrapping. There is no `SynthesisDefinition` bundle because there is
only one synthesis node.

### TokenManager

A per-API component responsible for obtaining and caching a bearer token from the
shared OAuth token server. Tokens are cached for a fixed TTL (15 minutes).
Each static source has its own `TokenManager` backed by its own credentials
(username/password from environment variables).

### HTTP Client

An `httpx.AsyncClient` instance, constructed at graph build time. The graph owns
six clients in total:

- One per Static Source (4 total), each bound to that source's base URL and
  injected into the source's fetch node via closure.
- One for the LLM Gateway, injected into the synthesis node.
- One shared OAuth client, injected into every `TokenManager`. All TokenManagers
  hit the same OAuth token server, so one connection pool is sufficient.

Clients are constructed once at startup and reused across invocations.

### Node Factory

A function that accepts a `Source Definition`, an HTTP client, and a `TokenManager`,
and returns an async LangGraph node function. A single factory plus one `Source
Definition` per source replaces having four near-identical per-source factory
functions. Enables isolated unit testing by accepting fakes at construction time.

### Graph State

A flat Pydantic model with one typed field per static source plus the synthesis
output. Each fetch node writes exactly one field. The synthesis node reads all
source fields and writes the final output field.

### Source Model

A Pydantic model representing the parsed, validated result of one static source's
API response. Each static source has its own `SourceModel`. Parsing happens inside
the fetch node — raw HTTP responses never appear in state.

### Source Definition

The bundle of facts that distinguish one Static Source from the others: its name,
its `SourceModel`, how to build its request, and how to parse its response. Each
Static Source has exactly one `Source Definition`, declared at design time. The
`Node Factory` consumes a `Source Definition` to produce a fetch node, so all
source-specific logic lives in one value per source rather than being scattered
across separate factory functions.

The two callables are **pure functions**:

- `build_request(state) -> RequestSpec` returns a transport-agnostic value
  object (`method`, `path`, optional `json` / `params` / `headers`). It does
  not see the bearer token; the factory injects `Authorization` and `fsreqid`
  headers after the spec is built.
- `parse_response(response) -> SourceModel` takes the raw `httpx.Response`
  (not pre-parsed JSON), so a source can inspect status codes or headers when
  it needs to.

Keeping both pure means each source's request shape and response parsing are
testable without a network, a TokenManager, or any factory plumbing.

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

The caller-supplied identifier for a graph invocation. Used in three places on
every run, with the same value:

- In `GraphState.fs_req_id`, where fetch nodes and the synthesis node read it.
- As the LangGraph `thread_id` in `config["configurable"]["thread_id"]`, which
  scopes the checkpoint to a single request.
- As the `fsreqid` request header on every upstream HTTP call (each Static
  Source and the LLM Gateway), so operators can trace a graph run end-to-end.

The recommended entry point is the `run(graph, fs_req_id, ...)` helper, which
accepts `fs_req_id` once and constructs both the `GraphState` input and the
LangGraph config, so the caller cannot put a different value in each place.
Calling `graph.ainvoke` directly is still possible but skips the invariant.

### SourceFetchError

A typed exception raised by a fetch node when its upstream API call fails.
Carries `source: str` (the source name) and `cause: Exception`.
The graph aborts and propagates this exception to the caller — synthesis is skipped.
The caller is responsible for catching `SourceFetchError` and handling it.

A run that aborted is **dead**: recovery is "fail and restart" — the caller mints
a new `fs_req_id` for the retry. There is no resume-from-checkpoint, and the
abort behavior is not specified to preserve partial state.
