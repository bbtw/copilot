# Retirement Plan Optimizer

A LangGraph pipeline that optimizes a customer's retirement savings strategy. Given a customer profile, it determines the optimal annual contribution amounts across account types to maximize **after-tax projected wealth** at retirement, then generates a plain-English explanation using three parallel LLM analysis branches.

## What it does

1. **Loads a customer profile** — age, income, expenses, existing balances, employer match, tax rates
2. **Runs a cvxpy LP optimizer** — finds the optimal annual contribution allocation across six account types, in two phases (pre-50 and post-50 401k/IRA catch-up)
3. **Runs a Monte Carlo simulation** — 1,000 paths varying market returns and inflation; reports percentiles in today's dollars and scores retirement readiness
4. **Fans out to three parallel LLM branches:**
   - **Standard analysis** — narrates what the optimizer selected and why
   - **Accumulation analysis** — uses local LP sensitivity probes, then sends the analysis prompt through the LLM gateway
   - **Withdrawal analysis** — uses local Monte Carlo sensitivity probes, then sends the analysis prompt through the LLM gateway
5. **Synthesizes a final explanation** — fourth LLM call combining all three analyses, with tone shaped by confidence band
6. **Formats and returns the result** — allocation, projected wealth, percentile distribution, confidence score, explanation

## Account types

| Account | Tax treatment |
|---|---|
| 401k | Pre-tax contributions, taxed at withdrawal |
| Roth 401k | Post-tax contributions, tax-free at withdrawal |
| Traditional IRA | Pre-tax (modeled as fully deductible in v1) |
| Roth IRA | Post-tax (subject to income eligibility limit) |
| HSA | Triple tax-advantaged (requires HDHP enrollment) |
| Taxable brokerage | No limits, no tax advantages — overflow bucket |

## Setup

```bash
uv sync
```

Set environment variables (copy `.env.example` and adjust):

```bash
export LLM_GATEWAY_BASE_URL=http://127.0.0.1:8001/v1  # local gateway shim
export LLM_MODEL_ID=llama3.2
# Production: point at the enterprise gateway URL and set LLM_MODEL_PROVIDER

# Optional readable console logging
export LOG_LEVEL=INFO
export LOG_COLOR=auto  # auto, always, or never

# Optional LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
```

For local LLM runs, start Ollama first. The run script starts the FastAPI gateway explicitly with `uvicorn`.

```bash
ollama serve
```

The shim forwards non-streaming OpenAI-compatible `/v1/chat/completions` requests to Ollama's `/v1/chat/completions` endpoint. Override `LOCAL_LLM_UPSTREAM_BASE_URL`, `LOCAL_LLM_UPSTREAM_API_KEY`, `LOCAL_LLM_GATEWAY_HOST`, or `LOCAL_LLM_GATEWAY_PORT` only when the defaults do not match your local setup.

## Run

```bash
./scripts/run-retirement-plan.sh
```

The wrapper script restarts the FastAPI LLM gateway before running the graph. Gateway logs stream to the terminal by default. To also write them to a file, set `LOCAL_LLM_GATEWAY_LOG_FILE`.

## Testing

```bash
uv run pytest
```

Tests cover the LP optimizer constraints, Monte Carlo projection behavior, explanation service prompt assembly, and node wiring. Tests do not require an LLM API key.

## Project structure

```
├── src/
│   └── lang_graph_state/
│       ├── main.py                       ← builds and runs the LangGraph graph
│       ├── instrumentation/
│       │   ├── nodes.py                  ← node timing/logging wrapper
│       │   └── tracing.py                ← RunnableConfig tags, metadata, and thread_id helper
│       ├── domain/
│       │   ├── state.py                  ← RetirementPlanState Pydantic model + reducer-backed analysis_sections
│       │   └── models.py                 ← domain models: CustomerProfile, ContributionAllocation, PlanAnalysisSection, etc.
│       ├── services/
│       │   ├── lp_solver.py              ← LP solve logic; public interface: solve_lp(profile) -> dict
│       │   ├── mc_simulator.py           ← Monte Carlo simulation; public interface: simulate(profile, allocation) -> dict
│       │   ├── lp_sensitivity.py         ← LP sensitivity probes for accumulation analysis
│       │   ├── mc_sensitivity.py         ← MC sensitivity probes for withdrawal analysis
│       │   ├── explanation.py            ← prompt assembly and LLM orchestration for all analysis text
│       │   └── llm.py                    ← GatewayClient; OpenAI-format REST API wrapper
│       ├── llm_gateway/
│       │   └── server.py                 ← local FastAPI gateway shim for Ollama-compatible development
│       └── nodes/
│           ├── load_profile.py           ← mocked CustomerProfile
│           ├── lp_optimizer.py           ← thin wrapper: calls solve_lp(); sets contribution_allocation + projected_wealth
│           ├── monte_carlo.py            ← thin wrapper: calls simulate(); sets wealth_distribution + confidence_score + confidence_band
│           ├── analysis_agents.py        ← accumulation and withdrawal analysis node builders
│           ├── standard_analysis.py      ← thin wrapper: writes the standard PlanAnalysisSection
│           ├── synthesize_explanation.py ← thin wrapper: awaits ExplanationService.asynthesize()
│           └── format_output.py          ← assembles RetirementPlanResult from state
└── tests/
```

## How state flows through the graph

The graph is a `StateGraph(RetirementPlanState)` run with `ainvoke()`. Each node receives the accumulated Pydantic state object and returns only the keys it owns as `dict[str, Any]`. Calculation nodes are sync; LLM-calling nodes are async and await the gateway-backed client. LangGraph merges each partial dict before calling the next node.

Most fields use the default reducer because exactly one node owns each field. The exception is `analysis_sections`: all three parallel LLM branches write one `PlanAnalysisSection` to that same key, so it uses an explicit reducer (`merge_analysis_sections`) to merge concurrent writes deterministically and dedupe by section kind.

| Node | Reads | Returns |
|---|---|---|
| `load_customer_profile` | initial empty state | `customer_profile` |
| `run_lp_optimizer` | `customer_profile` | `contribution_allocation`, `projected_wealth`, `optimization_diagnostics` |
| `run_monte_carlo` | `customer_profile`, `contribution_allocation` | `wealth_distribution`, `confidence_score`, `confidence_band` |
| `run_standard_analysis` *(parallel)* | `customer_profile`, `contribution_allocation`, `projected_wealth`, `wealth_distribution`, `confidence_score`, `confidence_band` | `analysis_sections[]` with `kind="standard"` |
| `run_accumulation_agent` *(parallel)* | `customer_profile`, `contribution_allocation`, `projected_wealth`, `optimization_diagnostics` | `analysis_sections[]` with `kind="accumulation"` |
| `run_withdrawal_agent` *(parallel)* | `customer_profile`, `wealth_distribution`, `confidence_score`, `confidence_band` | `analysis_sections[]` with `kind="withdrawal"` |
| `synthesize_explanation` | `analysis_sections`, `confidence_band` | `explanation` |
| `format_output` | all prior outputs | `result` |

The pipeline fans out from `run_monte_carlo` to all three analysis nodes simultaneously and fans back in at `synthesize_explanation`. There are no conditional edges; `confidence_band` is data passed to the synthesis prompt, not a routing decision.

## Key design decisions

- **Two-phase LP** — separate allocation vectors for pre-50 and post-50 years, capturing age-50 catch-up contribution limits for 401k and IRA accounts
- **Deep modules, simple interfaces** — `src/lang_graph_state/services/lp_solver.py` and `src/lang_graph_state/services/mc_simulator.py` own the calculation logic; node wrappers are 5–12 lines calling into those services
- **After-tax objective** — pre-tax accounts scaled by `(1 − retirement_tax_rate)` so the optimizer correctly favors Roth accounts when current rates exceed retirement rates
- **MC percentiles in today's dollars** — terminal wealth deflated by simulated inflation for interpretable percentiles; confidence measures whether assets can fund planned retirement expenses after expected retirement income
- **Confidence band is data, not routing** — all three analysis branches always run; `confidence_band` shapes synthesis tone rather than branching the graph
- **Reducer-backed parallel state** — the three parallel branches write to `analysis_sections` through an explicit reducer instead of relying on independent ad hoc fields
- **Gateway-only LLM access** — every graph LLM branch awaits `ExplanationService`, which uses `GatewayClient`; local LP and MC probes are calculated before the gateway call and included in the prompt
- **Instrumentation folder** — graph node timing lives in `instrumentation/nodes.py`; invoke config, LangSmith tags, metadata, and checkpoint `thread_id` setup live in `instrumentation/tracing.py`
- **Async LLM path** — LLM-calling graph nodes await `ExplanationService`, while the local gateway shim proxies upstream with async `httpx`
- **Injectable LLM and service** — `ExplanationService` accepts a `GatewayClient` so tests can mock LLM access without patching imports
- **Monte Carlo RNG is injected** — `build_monte_carlo_node(rng=None)` accepts an optional RNG; tests pass `np.random.default_rng(42)` for determinism

See `CONTEXT.md` for the full domain glossary and architecture decisions.

For an interactive walkthrough of the graph flow, open `docs/graph-explorer.html` in a browser.

## Out of scope

- Multi-year LP with income growth and IRS limit indexing
- Social Security benefit calculation (expected retirement income is supplied as profile data)
- Per-account differential returns
- Roth conversions, backdoor Roth
- State income tax
- Conversational input gathering (LLM-driven profile collection)
- Interactive low-confidence revision workflow (PlanRevisionIntake, RevisedPlanScenario)
