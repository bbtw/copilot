# lang-graph-state

A bare-bones LangGraph template that demonstrates the patterns worth getting right before any real domain logic is wired in:

- sequential nodes with a single owner per state field
- a conditional fan-out (`add_conditional_edges`) that picks branches at runtime
- concurrent writes to a shared list field, merged by an explicit reducer
- a fan-in node that consumes the merged sections
- thin node wrappers over a separate services layer
- a single async LLM gateway (`GatewayClient`) injected into the graph and mocked in tests
- LangSmith tracing via `langsmith.wrappers.wrap_openai`, opt-in by env var

Domain models (`Profile`, `LPResult`, `MCResult`) are intentionally empty `BaseModel`s. The point of v1 is the graph shape; field schemas and real solvers/simulators get wired in later without changing node signatures.

## Graph

```
START
  ↓
load_profile
  ↓
run_lp_optimizer
  ↓
run_monte_carlo
  ↓ (conditional fan-out — route_after_mc decides which branches run)
┌────────────────────────┬───────────────────────────┬──────────────────────────┐
│ run_standard_explanation │ run_contribution_explanation │ run_withdrawal_explanation │
└────────────────────────┴───────────────────────────┴──────────────────────────┘
  ↓ (fan-in — reducer merges concurrent writes)
synthesize_explanation
  ↓
format_output
  ↓
END
```

`route_after_mc` in `src/lang_graph_state/nodes/routing.py` returns the list of branch node names to schedule. Today it returns all three; the file is the seam where real conditional logic (e.g. on `mc_result.confidence_score`) gets added.

## State

`GraphState` (`src/lang_graph_state/domain/state.py`) is a Pydantic model. Each node returns only the keys it owns; LangGraph merges them in. The single field with a reducer is `analysis_sections` because the three fan-out branches write it concurrently.

| Field | Type | Set by |
|---|---|---|
| `profile` | `Profile \| None` | `load_profile` |
| `lp_result` | `LPResult \| None` | `run_lp_optimizer` |
| `mc_result` | `MCResult \| None` | `run_monte_carlo` |
| `analysis_sections` *(reducer)* | `list[AnalysisSection]` | three fan-out branches |
| `explanation` | `str` | `synthesize_explanation` |
| `final_output` | `FinalOutput \| None` | `format_output` |

`merge_analysis_sections` dedupes by `kind` (a retried branch overwrites instead of appending) and sorts by `ANALYSIS_SECTION_ORDER` so the merged list is stable regardless of which branch finishes first.

## Project layout

```
src/lang_graph_state/
├── main.py                          ← build_graph() wires nodes + edges; amain() runs end-to-end
├── domain/
│   ├── state.py                     ← GraphState + reducer-backed analysis_sections
│   └── models.py                    ← Profile, LPResult, MCResult, AnalysisSection, FinalOutput, merge_analysis_sections
├── nodes/
│   ├── load_profile.py              ← sync, returns {"profile": Profile()}
│   ├── lp_optimizer.py              ← async, awaits solve_lp(profile)
│   ├── monte_carlo.py               ← async, awaits simulate(profile, lp_result)
│   ├── routing.py                   ← route_after_mc(state) -> list[str] of branch node names
│   ├── sections.py                  ← three async branch-node builders, each closes over a GatewayClient
│   ├── synthesize_explanation.py    ← async fan-in node builder; closes over a GatewayClient
│   └── format_output.py             ← sync, returns the final FinalOutput
├── services/
│   ├── gateway.py                   ← GatewayClient: AsyncOpenAI wrapped with langsmith
│   ├── lp_solver.py                 ← async stub: solve_lp(profile) -> LPResult
│   └── mc_simulator.py              ← async stub: simulate(profile, lp_result) -> MCResult
└── instrumentation/
    ├── logging.py                   ← HumanReadableFormatter, configure_logging()
    └── tracing.py                   ← build_invoke_config(thread_id) for checkpointing
tests/
├── test_main.py                     ← build_graph + end-to-end with a mocked GatewayClient
└── test_state.py                    ← reducer dedup + canonical ordering
docs/
└── graph.html                       ← static graph + state reference page
```

## Setup

```bash
uv sync
```

Required env vars:

```bash
LLM_GATEWAY_BASE_URL=http://127.0.0.1:8001/v1   # any OpenAI-compatible /v1 endpoint
LLM_GATEWAY_API_KEY=...                          # optional; defaults to empty string
LLM_MODEL_ID=qwen2.5:1.5b
```

Optional:

```bash
LOG_LEVEL=INFO          # passed to configure_logging()
LOG_COLOR=auto          # auto | always | never (NO_COLOR=1 also disables color)

LANGSMITH_TRACING=true  # wrap_openai becomes a real tracer when LANGSMITH_API_KEY is set
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

`main.py` calls `load_dotenv()` at import time, so a `.env` file in the project root is picked up automatically.

## Run

```bash
uv run python -m lang_graph_state.main
```

`amain()` builds the graph with a `MemorySaver` checkpointer, generates a UUID `thread_id`, and runs a single `ainvoke({})`. Any OpenAI-compatible `/v1/chat/completions` endpoint works at `LLM_GATEWAY_BASE_URL` — point at an Ollama-backed shim, an enterprise gateway, or a hosted provider.

## Tests

```bash
uv run pytest
```

Tests mock `GatewayClient` with `MagicMock(spec=GatewayClient)` and an `AsyncMock` for `acomplete`, so no LLM endpoint is needed.

## Where things plug in

- **Real domain shape** — extend `Profile`, `LPResult`, `MCResult` in `domain/models.py`. Node signatures don't change.
- **Real LP / Monte Carlo** — replace the stubs in `services/lp_solver.py` and `services/mc_simulator.py`. Nodes call them through a stable async interface.
- **Real conditional routing** — edit only `nodes/routing.py`; topology and node logic stay untouched.
- **New analysis branch** — add a `kind` to `AnalysisKind`, an entry in `ANALYSIS_SECTION_ORDER`, a builder in `nodes/sections.py`, and wire it in `routing.py` + `main.py`.

See `docs/graph.html` for a static one-page reference, and `CONTEXT.md` for the rationale behind each architectural choice.
