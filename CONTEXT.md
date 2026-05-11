# lang-graph-state — Context

## Purpose

A minimal LangGraph template that locks in the graph-shape decisions worth getting right before any real domain logic exists: single-writer state with one reducer-backed shared field, conditional fan-out, fan-in, and a single injected async LLM gateway. Domain models and services are stubs; the value of v1 is the topology and the seams.

## Pipeline

```
load_profile
      ↓
run_lp_optimizer        ← awaits solve_lp(profile) -> LPResult
      ↓
run_monte_carlo         ← awaits simulate(profile, lp_result) -> MCResult
      ↓ (conditional fan-out via route_after_mc)
┌───────────────────────────────────────────────────────────────────────┐
│ run_standard_explanation │ run_contribution_explanation │ run_withdrawal_explanation │
└───────────────────────────────────────────────────────────────────────┘
      ↓ (fan-in — analysis_sections reducer dedupes + sorts)
synthesize_explanation
      ↓
format_output
```

| Node | Reads | Returns |
|---|---|---|
| `load_profile` | (initial empty state) | `profile` |
| `run_lp_optimizer` | `profile` | `lp_result` |
| `run_monte_carlo` | `profile`, `lp_result` | `mc_result` |
| `run_standard_explanation` *(branch)* | (state) | `analysis_sections[]` with `kind="standard_explanation"` |
| `run_contribution_explanation` *(branch)* | (state) | `analysis_sections[]` with `kind="contribution_explanation"` |
| `run_withdrawal_explanation` *(branch)* | (state) | `analysis_sections[]` with `kind="withdrawal_explanation"` |
| `synthesize_explanation` | `analysis_sections` (post-merge) | `explanation` |
| `format_output` | `explanation` | `final_output` |

Calculation nodes (`run_lp_optimizer`, `run_monte_carlo`) are async because they await placeholder service coroutines. The three explanation branches and `synthesize_explanation` are async because they call `GatewayClient.acomplete`. `load_profile` and `format_output` are sync.

## Glossary

**GraphState**
Pydantic model defined in `src/lang_graph_state/domain/state.py`. Every node receives the full state and returns only the keys it owns; LangGraph merges them into the next state. All fields use the default last-write reducer except `analysis_sections`, which has an explicit reducer.

**Profile / LPResult / MCResult**
Placeholder domain models in `src/lang_graph_state/domain/models.py`. Empty `BaseModel`s in v1. Real field shapes get added when the upstream profile API and the LP / Monte Carlo services produce real results. Node signatures should not need to change.

**AnalysisSection**
The unit written by each branch in the fan-out. Carries a `kind` (`standard_explanation`, `contribution_explanation`, or `withdrawal_explanation`) and a `content` string. The reducer keys on `kind`.

**AnalysisKind / ANALYSIS_SECTION_ORDER**
`AnalysisKind` is the `Literal` enum of valid section kinds. `ANALYSIS_SECTION_ORDER` is the canonical ordering used by the reducer to stabilise the merged list regardless of which branch finished first. Adding a new branch means: add a `kind` here, add an order index, add a builder in `nodes/sections.py`, and wire it in `routing.py` and `main.py`.

**merge_analysis_sections**
The LangGraph reducer for `GraphState.analysis_sections`. Dedupes by `kind` (so a retried branch overwrites rather than appends) and returns the merged list sorted by `ANALYSIS_SECTION_ORDER`.

**route_after_mc**
The conditional-edge function attached to `run_monte_carlo` in `main.build_graph`. Receives the current `GraphState` and returns a `list[str]` of branch node names to schedule. Today it always returns all three; real conditional logic (e.g. inspecting `mc_result`) belongs here and nowhere else. LangGraph ignores outgoing edges of branches not returned from this function.

**GatewayClient**
The single concrete LLM client in `src/lang_graph_state/services/gateway.py`. Wraps `AsyncOpenAI` with `langsmith.wrappers.wrap_openai`. Reads `LLM_GATEWAY_BASE_URL`, `LLM_GATEWAY_API_KEY`, `LLM_MODEL_ID` from the environment. Exposes one async method, `acomplete(prompt, *, system, max_tokens)`. Injected into `build_graph` so tests can pass `MagicMock(spec=GatewayClient)` with an `AsyncMock` for `acomplete`.
_Avoid_: Constructing model clients inside nodes; provider-specific SDK imports outside this module; multiple adapter classes; switching logic in application code.

**solve_lp / simulate**
Async stubs in `services/lp_solver.py` and `services/mc_simulator.py`. Each awaits `asyncio.sleep(0)` and returns an empty result model. They exist to fix the node-to-service contract before real implementations are written.

**configure_logging**
Sets up a single console handler with `HumanReadableFormatter`. Reads `LOG_LEVEL` (default `INFO`) and `LOG_COLOR` (`auto` | `always` | `never`; `NO_COLOR=1` forces off). Called by `amain` so a fresh process gets readable logs without extra setup.

**build_invoke_config**
Builds the `RunnableConfig` passed to `app.ainvoke`. The only key it sets is `configurable.thread_id`, which the checkpointer uses to scope persisted state to a thread. Centralising this means run-name, tag, and metadata changes don't touch `main`.

**FinalOutput**
The terminal artifact written by `format_output`. Today it just carries `explanation: str`. It exists as a distinct contract so the graph has a clean "this is what the pipeline produced" handle separate from intermediate state.

## Architecture decisions

- **Single owner per state field, except for the shared list.** Last-write-wins is the default reducer. The only exception is `analysis_sections`, written by three branches in parallel — it gets an explicit reducer rather than three separate one-off fields per branch.

- **Reducer dedupes by `kind` and sorts canonically.** Idempotent merges (retries overwrite) and stable order across runs (synthesis sees the same input shape regardless of branch completion order).

- **Conditional fan-out, not always-on fan-out.** `add_conditional_edges("run_monte_carlo", route_after_mc)` instead of unconditional edges. Today the routing function returns all three branches, but the seam exists so adding real conditional logic doesn't require touching the topology or the node code. LangGraph only waits for the branches that were scheduled before running `synthesize_explanation`.

- **Routing lives in one file.** `nodes/routing.py` is the only place that needs to change when branching conditions evolve. Nodes don't make routing decisions, and `build_graph` doesn't either.

- **Thin nodes over deep services.** Each node is a handful of lines that calls into the service layer and returns the keys it owns. LP, MC, and LLM details live in `services/`; nodes contain no calculation or prompt logic.

- **Async LLM, sync calculation surface.** `GatewayClient` exposes only `acomplete`; all four LLM-calling nodes await it. Calculation nodes are async only because their service stubs are async — easy to keep async-first once the real implementations land.

- **Dependency injection for the gateway.** `build_graph(client: GatewayClient, *, checkpointer=None)` takes both the gateway and the checkpointer as parameters. Tests pass a `MagicMock(spec=GatewayClient)` with an `AsyncMock` for `acomplete`; no patching of imports, no environment setup required for tests.

- **LangSmith tracing is opt-in and lives at the gateway.** `wrap_openai` wraps the client at construction time. It's a no-op without `LANGSMITH_API_KEY`, and a real tracer with it. No tracing code in nodes or services.

- **Instrumentation is isolated from graph topology.** `instrumentation/logging.py` handles the console formatter; `instrumentation/tracing.py` builds the invoke config. Neither is imported by nodes or services.

- **Mock data sources first.** Profile, LPResult, MCResult are stubs; the real shape comes when the upstream APIs are wired in. The graph topology and node signatures shouldn't need to move when that happens.

- **Checkpointer is injected.** `build_graph` takes an optional `checkpointer`; `amain` passes `MemorySaver` and a UUID `thread_id`. Production swap (e.g. SQLite, Postgres) is a one-line change at the call site.

## Project layout

```
src/lang_graph_state/
├── main.py                          ← build_graph(), amain()
├── domain/
│   ├── state.py                     ← GraphState
│   └── models.py                    ← Profile, LPResult, MCResult, AnalysisSection, FinalOutput, AnalysisKind, ANALYSIS_SECTION_ORDER, merge_analysis_sections
├── nodes/
│   ├── load_profile.py
│   ├── lp_optimizer.py
│   ├── monte_carlo.py
│   ├── routing.py                   ← route_after_mc
│   ├── sections.py                  ← build_{standard,contribution,withdrawal}_explanation_node
│   ├── synthesize_explanation.py    ← build_synthesize_explanation_node
│   └── format_output.py
├── services/
│   ├── gateway.py                   ← GatewayClient
│   ├── lp_solver.py                 ← solve_lp (stub)
│   └── mc_simulator.py              ← simulate (stub)
└── instrumentation/
    ├── logging.py                   ← HumanReadableFormatter, configure_logging
    └── tracing.py                   ← build_invoke_config
tests/
├── test_main.py
└── test_state.py
docs/
└── graph.html
```

## Out of scope (v1)

- Real domain modelling on `Profile`, `LPResult`, `MCResult`.
- Real LP solver and Monte Carlo simulator implementations behind `solve_lp` and `simulate`.
- Non-trivial conditional routing — `route_after_mc` is the seam, but it returns all branches today.
- Human-in-the-loop / interrupt patterns. No `interrupt` calls; the graph runs straight through.
- Persistent checkpointers. `amain` uses `MemorySaver`; production wiring (SQLite/Postgres) is deferred.
- Prompt engineering. `acomplete` is called with empty strings as placeholders — real prompts come with the real domain.

## Project goal

Build v1 to best-practice LangGraph quality — idiomatic graph shape, clean state contracts, testability without an LLM endpoint. v2 features (real domain logic, real probes, interrupts, revision workflows) are explicitly out of scope until v1 is solid.
