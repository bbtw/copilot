# Retirement Plan Optimizer — Domain Context

## Purpose
A LangGraph pipeline that optimizes a customer's retirement savings strategy for a large financial organization (Vanguard-scale). Given a customer profile, it determines the optimal annual contribution amounts across account types to maximize **after-tax** projected wealth at retirement.

## Glossary

**CustomerProfile**
All inputs describing a customer's financial situation:
- `age`, `retirement_age`
- `annual_income`, `annual_expenses`
- `retirement_annual_expenses`, `retirement_years_to_plan`
- `expected_retirement_income` (v2; retirement income available outside portfolio withdrawals)
- Current balances per `AccountType`
- `employer_match_rate` (e.g., 0.50) and `employer_match_cap` (e.g., 0.06 of salary)
- `hdhp_enrolled` (HSA eligibility)
- `current_marginal_tax_rate`
- `assumed_retirement_marginal_tax_rate`
- `filing_status` (drives Roth IRA phase-out preprocessing)

Mocked for v1; replaced by real API data in future versions.

**AccountType**
One of six tax-treatment buckets the optimizer allocates contributions across:
- `401k` — employer-sponsored, pre-tax (taxed at withdrawal)
- `roth_401k` — employer-sponsored, post-tax (tax-free at withdrawal)
- `traditional_ira` — individual, pre-tax (modeled as fully deductible in v1, taxed at withdrawal)
- `roth_ira` — individual, post-tax (tax-free at withdrawal, subject to income eligibility limit)
- `hsa` — triple tax-advantaged, requires HDHP enrollment
- `taxable_brokerage` — no contribution limits, no tax advantages; overflow bucket (capital gains tax simplification: assume taxed at the retirement marginal rate)

**ContributionAllocation**
The LP optimizer's output: two allocation vectors (pre-50 and post-50), each a mapping of `AccountType` → annual contribution amount in dollars. For customers already 50+, only the post-50 vector is used. Asset allocation within accounts is not a decision variable.

**ExpectedReturn**
A single portfolio-level expected nominal return (default: 7%) applied uniformly across all account types in the LP's deterministic projection. Per-account differential returns are out of scope for v1. The Monte Carlo engine introduces variability around this mean using `return_std_dev = 0.15` (annualized). Inflation is modeled with `inflation_mean = 0.03` and `inflation_std_dev = 0.01`. These are system-level constants in v1.

**IRS Contribution Limits**
Hard upper bounds on annual contributions per `AccountType`, enforced as LP constraints. Includes age-50 catch-up contribution increases for 401(k) and IRA accounts. HSA catch-up begins at age 55, which v1's two-phase age-50 allocation does not model; v1 therefore uses the base HSA limit in both phases. Roth IRA limits are precomputed from `annual_income` and `filing_status` *before* the LP runs (input preprocessing of phase-out tapers). Traditional IRA deductibility phase-outs are out of scope in v1.

**EmployerMatch**
A subsidy on 401(k) and Roth 401(k) contributions: matched dollars increase the customer's wealth without coming from `SavingsCapacity`. The match cap applies to **combined** 401k + Roth 401k elective deferrals (real-world standard — one auxiliary variable in the LP). Modeled as `matched_dollars = match_rate × min(401k_contrib + roth_401k_contrib, match_cap × annual_income)`, decomposed into the standard piecewise-linear LP formulation. Matched dollars are added to the objective alongside customer contributions.

**SavingsCapacity**
The customer's total available dollars per year for retirement contributions (`annual_income − annual_expenses`). The binding budget constraint in the LP. Employer match dollars do not draw from this pool.

**ProjectedWealth**
The objective being maximized: estimated **after-tax** portfolio value at retirement age. For each account, terminal balance = `existing_balance × (1+r)^years + pre50_contrib × FV_annuity(r, pre50_years) + post50_contrib × FV_annuity(r, post50_years)`, where `FV_annuity(r, n) = ((1+r)^n − 1) / r`. After-tax scaling: Roth 401k, Roth IRA, HSA contribute full value; Traditional 401k and IRA are scaled by `(1 − retirement_tax_rate)`; taxable brokerage uses a simplified haircut at `retirement_tax_rate` (principal + gains both scaled — a conservative approximation). The LP objective is the sum of after-tax terminal balances across all six account types.

**OptimizationDiagnostics**
Optimizer-provided explainability output that describes why the selected allocation was chosen, including selected variables, binding constraints, and zero-selected variables with reduced-cost signals.
_Avoid_: Treating these diagnostics as cvxpy-specific debugging internals or coupling explanation behavior to the local v1 optimizer implementation

**MonteCarloProjection**
A simulation of 1,000+ retirement paths run after the LP produces a `ContributionAllocation`. Varies market returns and inflation stochastically. Includes employer match dollars using the same combined 401(k) + Roth 401(k) cap as the LP. Returns a `WealthDistribution` reported in **today's dollars** (deflated by simulated inflation), a `ConfidenceScore` that measures retirement readiness, and the derived `ConfidenceBand`.

**Production Simulation**
This repository simulates the shape of the production retirement planning pipeline so non-optimization concerns can be tuned before real planner integrations exist. In v1, the local LP optimizer and Monte Carlo engine are behavioral stand-ins rather than authoritative financial calculators; exact numerical correctness is secondary to stable contracts, realistic enough outputs, and clear graph/service boundaries. In production, the LP optimizer and Monte Carlo engine are separate black-box API calls. The only shared assumption is that both calls receive the same customer and plan input details needed for their calculations. They must not share calculation implementation code.

**WealthDistribution**
The spread of projected after-tax portfolio values at retirement across all Monte Carlo simulation paths (10th / 50th / 90th percentiles).
_Avoid_: Using this outside the Monte Carlo processor to make plan-quality decisions or route workflow

**RetirementOutcomeDistribution**
A v2 Monte Carlo result that describes retirement-readiness outcomes across simulated paths as shortfall/surplus percentiles after funding `RetirementAnnualExpenses` for `RetirementYearsToPlan` years.
Customer-facing output may show this distribution as secondary evidence for the `ConfidenceScore`, but graph routing and plan selection must not depend on its percentiles.

**ConfidenceScore**
The probability that the plan can fund `RetirementAnnualExpenses` for `RetirementYearsToPlan` years after the customer's retirement age without assets being exhausted before the end of the planned retirement horizon. The Monte Carlo processor owns this calculation and returns the score; it answers "how likely is this customer to have enough money to retire under the simulated paths?"

**ConfidenceBand**
A data field derived from `ConfidenceScore` by `classify_confidence()` in `src/lang_graph_state/domain/models.py`; canonical values are `low`, `medium`, and `high`. Passed as an input to the synthesis LLM call to shape tone — not used to branch the graph. Client-facing explanations should translate it into softer projection-risk language rather than presenting the raw band label as a plan quality judgment.
_Avoid_: Customer-facing headline labels like "low confidence plan"; conditional graph edges based on this field

Initial thresholds:
- `low`: `ConfidenceScore < 70%`
- `medium`: `70% ≤ ConfidenceScore < 85%`
- `high`: `ConfidenceScore ≥ 85%`

These thresholds are calibrated against retirement readiness: whether retirement income and assets cover `RetirementAnnualExpenses` for `RetirementYearsToPlan` years after the customer's retirement age.

Explanation behavior:
- `high`: Explain the allocation primarily as the optimizer's selected strategy, mention uncertainty briefly, and emphasize the major drivers such as employer match, HSA eligibility, tax treatment, and binding constraints.
- `medium`: Explain the allocation and add a clear uncertainty paragraph: the plan may meet the retirement-readiness goal, but outcomes remain sensitive to market, inflation, and spending paths, so the customer should revisit the plan periodically.
- `low`: Do not present the plan as likely to meet the customer's retirement goal. Explain the allocation, state that simulated paths often exhaust assets before the planned retirement horizon, and suggest review levers such as increasing savings capacity, changing retirement age, reducing expenses, or advisor review. Avoid saying the allocation is "bad"; the issue is retirement-readiness risk.

**RetirementYearsToPlan**
The number of retirement years the plan must fund after the customer's retirement age.
_Avoid_: Using life expectancy or retirement end age as the canonical field name; either may inform the value, but the planning input is `RetirementYearsToPlan`

**RetirementAnnualExpenses**
The modeled annual spending need during retirement.
_Avoid_: Treating current `annual_expenses` as the canonical retirement spending amount; it may be used as a default only when a retirement-specific amount is unavailable

**ExpectedRetirementIncome**
Customer profile data representing a single annual amount of expected retirement income available outside portfolio withdrawals, such as Social Security estimates, pension income, annuity payments, or other recurring retirement income.
_Avoid_: Having the graph calculate Social Security benefits or pension values; modeling retirement income as a start/end-age stream in the first implementation

**PlanRevisionIntake**
A v2 low-confidence workflow step that presents model-informed candidate levers and asks the customer which future planning assumptions they are willing to change before running another optimization, limited to `retirement_age`, `RetirementAnnualExpenses`, `RetirementYearsToPlan`, and working-age expenses or savings targets that affect `SavingsCapacity`.
_Avoid_: Free-form "fix my plan" chat; mutating historical/current facts such as current age, existing balances, filing status, HDHP enrollment, current income, or employer match; unilaterally setting revised assumptions without customer approval

**RevisedPlanScenario**
A customer-approved set of changed planning assumptions produced by `PlanRevisionIntake` and submitted as one coherent input for a subsequent optimization run.
_Avoid_: Rerunning optimization after each individual intake answer

**PlanRevisionPreferences**
The accepted and declined planning levers captured during `PlanRevisionIntake`.
_Avoid_: Recommending the same declined lever again in the revised-result explanation

**RevisionAttemptLimit**
The v2 rule that permits at most one automatic low-confidence `PlanRevisionIntake` and optimization rerun before presenting the best available result with next-step guidance.
_Avoid_: Unbounded advisory loops inside the graph

**BaselinePlanResult**
The original optimization and Monte Carlo result produced before any low-confidence revision intake.

**FinalPlanResult**
The plan result presented as primary output; when a customer approves a `RevisedPlanScenario`, the revised result becomes primary and the baseline remains comparison context.

**PlanResultComparison**
A comparison between baseline and revised plan results that ranks improvement by `ConfidenceScore` first and treats `ProjectedWealth` as secondary context.
_Avoid_: Treating higher projected wealth or `WealthDistribution` percentiles as better when retirement readiness worsens

**PlanningHorizon**
A fixed retirement age used as the endpoint of the accumulation phase in the Monte Carlo simulations. The first readiness implementation models retirement survival for `RetirementYearsToPlan` years after this age.

**PlanExplanationRequest**
The service-level contract consumed by `ExplanationService` to generate a client-facing explanation from optimizer and Monte Carlo result contracts.
_Avoid_: Passing raw `RetirementPlanState` into explanation services or coupling explanation generation to LangGraph state shape

**LPSensitivityProbe**
A local calculation used by the accumulation analysis branch. Reruns the LP solver (`src/lang_graph_state/services/lp_solver.py`) with a perturbed copy of the customer profile (savings capacity delta, retirement age shift, or employer match cap change) and returns the delta in projected wealth. Each probe is a full `solve_lp` call on a modified `CustomerProfile`; probes are fast because the LP is cheap to solve locally.
_Avoid_: Sharing LP implementation code with the production optimizer; mutating the original profile instead of using `model_copy(update={...})`

**MCSensitivityProbe**
A local calculation used by the withdrawal analysis branch. Performs a partial Monte Carlo rerun under a perturbed input (annual expense delta, retirement duration extension, or withdrawal rate query). Uses a fixed RNG seed for comparability across probe calls. Returns updated `ConfidenceScore` and `WealthDistribution` percentiles.
_Avoid_: Full 1,000-path reruns for every probe invocation; sharing MC implementation code with the production simulator

**StandardPlanAnalysis**
The plain-English overview produced by the first parallel LLM node. Narrates what the optimizer selected and why, referencing `OptimizationDiagnostics`, and describes the Monte Carlo result at a summary level. No tools; simple LLM completion.

**PlanAnalysisSection**
One normalized output from a parallel LLM branch. Each section has a `kind` (`standard`, `accumulation`, or `withdrawal`) and `content`. All three parallel branches write `PlanAnalysisSection` values to the shared `analysis_sections` state key, which is merged by an explicit LangGraph reducer.
_Avoid_: Separate one-off state fields for each branch when the graph is meant to demonstrate concurrent writes to shared state

**AccumulationAnalysisAgent**
An async LangGraph node that runs local LP sensitivity probes, then asks `ExplanationService` for gateway-backed accumulation analysis. Analyzes contribution sequencing, the employer match lever, the two-phase pre-50/post-50 allocation shift, and binding constraints. Produces a `PlanAnalysisSection(kind="accumulation")`.
_Avoid_: Constructing model clients in the node; all LLM access should go through `ExplanationService`

**WithdrawalAnalysisAgent**
An async LangGraph node that runs local Monte Carlo sensitivity probes, then asks `ExplanationService` for gateway-backed withdrawal analysis. Analyzes retirement drawdown sustainability, sequence-of-returns risk, and sensitivity of `ConfidenceScore` to changed spending or duration assumptions. Produces a `PlanAnalysisSection(kind="withdrawal")`.
_Avoid_: Constructing model clients in the node; all LLM access should go through `ExplanationService`

**AnalysisSynthesis**
The fourth LLM call in the pipeline. Receives the three `PlanAnalysisSection` values and `ConfidenceBand` as inputs and produces the single client-facing explanation. No tools; simple LLM completion. Confidence band shapes tone rather than routing.

**GatewayClient**
A single concrete class in `src/lang_graph_state/services/llm.py` that wraps an OpenAI-format REST API call. Configured via environment variables (base URL, model provider, model ID). Points at the enterprise LLM gateway in production and at the local FastAPI gateway shim for Ollama-backed development. The async graph path uses `GatewayClient.acomplete()` only through `ExplanationService`; the sync `complete()` method remains for focused service tests and compatibility. The OpenAI clients are wrapped with `langsmith.wrappers.wrap_openai` at construction time so every LLM call is traced in LangSmith when `LANGSMITH_TRACING=true`; the wrapper is a no-op when tracing is not configured. Tests mock it at the call site using `MagicMock(spec=GatewayClient)`.
_Avoid_: Direct model clients in graph nodes, provider-specific SDK imports outside this module, multiple adapter classes, switching logic in application code

**RetirementPlanResult**
The final pipeline output: `ContributionAllocation`, `ProjectedWealth`, `WealthDistribution`, `ConfidenceScore`, `ConfidenceBand`, and a plain-English explanation generated by the LLM. Returned as a structured dict with a print helper for human-readable summary.

**RetirementPlanState**
The accumulated LangGraph state passed through the retirement planning pipeline, containing each node's completed outputs. Most fields are single-writer last-value fields. `analysis_sections` is the reducer-backed shared field used by the parallel LLM fan-out.

## Pipeline (v2)

```
load_customer_profile
       ↓
run_lp_optimizer        ← maximizes after-tax ProjectedWealth; outputs ContributionAllocation
       ↓
run_monte_carlo         ← outputs WealthDistribution (today's dollars) + ConfidenceScore
      ↓ (fan-out — three analyses run in parallel)
┌─────────────────────────────────────────────────────────────────┐
│  run_standard_analysis  │  run_accumulation_agent  │  run_withdrawal_agent  │
│  (gateway LLM)          │  (LP probes + gateway)   │  (MC probes + gateway) │
└─────────────────────────────────────────────────────────────────┘
       ↓ (fan-in)
synthesize_explanation  ← fourth LLM call; receives all three analyses + ConfidenceBand
       ↓
format_output           ← mechanical assembly; no LLM
```

## Project Structure

```
lang-graph-state/
├── src/
│   └── lang_graph_state/
│       ├── main.py              ← builds and runs the LangGraph graph
│       ├── instrumentation/
│       │   ├── nodes.py         ← node timing/logging wrapper
│       │   └── tracing.py       ← RunnableConfig tags, metadata, and checkpoint thread_id helper
│       ├── domain/
│       │   ├── state.py         ← RetirementPlanState Pydantic model + analysis_sections reducer
│       │   └── models.py        ← domain models: CustomerProfile, ContributionAllocation, PlanAnalysisSection, etc.
│       ├── services/
│       │   ├── lp_solver.py     ← LP solve logic; public interface: solve_lp(profile) -> dict
│       │   ├── mc_simulator.py  ← Monte Carlo simulation; public interface: simulate(profile, allocation) -> dict
│       │   ├── lp_sensitivity.py ← LP sensitivity probes for accumulation analysis
│       │   ├── mc_sensitivity.py ← MC sensitivity probes for withdrawal analysis
│       │   ├── explanation.py   ← prompt assembly and LLM orchestration for all analysis text
│       │   └── llm.py           ← GatewayClient; OpenAI-format REST API wrapper
│       └── nodes/
│           ├── load_profile.py  ← mocked CustomerProfile (42yo, $120k income, HDHP enrolled)
│           ├── lp_optimizer.py  ← thin wrapper: calls solve_lp(); sets contribution_allocation + projected_wealth + optimization_diagnostics
│           ├── monte_carlo.py   ← thin wrapper: calls simulate(); sets wealth_distribution + confidence_score + confidence_band
│           ├── analysis_agents.py ← builds accumulation and withdrawal analysis nodes
│           ├── standard_analysis.py ← thin wrapper: writes the standard PlanAnalysisSection
│           ├── synthesize_explanation.py ← thin wrapper: awaits ExplanationService.asynthesize(); sets explanation
│           └── format_output.py ← assembles RetirementPlanResult from state
├── docs/
│   └── graph-explorer.html      ← interactive graph flow explorer
└── tests/
```

Each node is a callable over `(state: RetirementPlanState) -> dict[str, Any]`. Calculation nodes are sync; LLM-calling nodes are async. The graph starts with `app.ainvoke({})`, so the initial state is empty. Every node reads the attributes it requires from the accumulated Pydantic state object and returns only the keys it produces; LangGraph merges that partial dict into the state before invoking the next node. The three parallel analysis nodes all write `analysis_sections`, so that field uses `merge_analysis_sections` as an explicit reducer.

| Node | Reads | Returns |
|---|---|---|
| `load_customer_profile` | initial empty state | `customer_profile` |
| `run_lp_optimizer` | `customer_profile` | `contribution_allocation`, `projected_wealth`, `optimization_diagnostics` |
| `run_monte_carlo` | `customer_profile`, `contribution_allocation` | `wealth_distribution`, `confidence_score`, `confidence_band` |
| `run_standard_analysis` | `customer_profile`, `contribution_allocation`, `projected_wealth`, `wealth_distribution`, `confidence_score`, `confidence_band` | `analysis_sections[]` with `kind="standard"` |
| `run_accumulation_agent` | `customer_profile`, `contribution_allocation`, `projected_wealth`, `optimization_diagnostics` | `analysis_sections[]` with `kind="accumulation"` |
| `run_withdrawal_agent` | `customer_profile`, `wealth_distribution`, `confidence_score`, `confidence_band` | `analysis_sections[]` with `kind="withdrawal"` |
| `synthesize_explanation` | `analysis_sections`, `confidence_band` | `explanation` |
| `format_output` | all prior outputs | `result` |

## Architecture Decisions

- **LP → MC is sequential** (not iterative). The LP maximizes wealth deterministically; MC scores the resulting plan stochastically. Iterative feedback loop remains out of scope.
- **LP and MC are production black boxes**. This repo may simulate both implementations locally, but production will call separate LP optimizer and Monte Carlo APIs. Keep the shared seam at the input details and result contracts; do not factor LP and MC calculation rules into shared implementation code.
- **LP remains an accumulation optimizer**. The LP objective remains after-tax `ProjectedWealth`; Monte Carlo owns readiness scoring and decumulation-style survival analysis.
- **LP is multi-year, two-phase**. Solves for two allocation vectors: pre-50 and post-50, reflecting the IRS catch-up contribution increase at age 50. Account balances compound across both phases; income, expenses, and tax rates are held fixed. Income growth and IRS limit indexing are deferred.
- **Objective is after-tax wealth**, using `assumed_retirement_marginal_tax_rate` to scale pre-tax balances. Without this, the LP would always favor Traditional accounts (wrong answer).
- **Asset allocation is fixed** — a single `ExpectedReturn` applied uniformly. Per-account differential returns deferred.
- **Employer match is modeled as a subsidy** in the objective (piecewise-linear `min`), not a constraint.
- **Roth IRA phase-outs are input preprocessing**, not LP constraints — collapsed to fixed limits before the LP runs. Traditional IRA deductibility phase-outs are deferred.
- **MC reports distribution percentiles in today's dollars** by deflating with simulated inflation. Confidence is returned by the Monte Carlo processor as the probability that retirement income and assets cover `RetirementAnnualExpenses` for `RetirementYearsToPlan` years after the customer's retirement age.
- **Three parallel LLM branches run after Monte Carlo.** The parent graph fans out from `run_monte_carlo` to `run_standard_analysis`, `run_accumulation_agent`, and `run_withdrawal_agent` simultaneously; LangGraph executes them in parallel. Each writes one `PlanAnalysisSection` to the shared reducer-backed `analysis_sections` field. A fourth `synthesize_explanation` node fans them back in.
- **LLM branches are async.** The parent graph is invoked with `ainvoke()`. All text-generation branches await `ExplanationService`, which calls `GatewayClient.acomplete()`. Sync calculation nodes remain sync.
- **Confidence band is data, not routing.** There are no conditional edges based on `ConfidenceBand`. All three analysis branches always run; the synthesis node receives `ConfidenceBand` as an input and uses it to shape tone. This replaces the previous `route_by_confidence_band` conditional edge.
- **Accumulation and withdrawal analyses run probes before the LLM call.** The nodes run deterministic local probes, then include probe outputs in the prompt passed through `ExplanationService`.
- **Accumulation probes are LP sensitivity probes** — read-only recalculations of LP outcomes for changed inputs (savings capacity delta, retirement age shift, employer match cap change).
- **Withdrawal probes are MC sensitivity probes** — MC reruns for changed inputs (annual expense delta, retirement duration extension, withdrawal rate query). Probes use a fixed RNG seed for comparability.
- **Synthesis node receives all three analysis sections plus `ConfidenceBand`** and produces the single client-facing explanation. It is a simple LLM completion (no tools, no loop).
- **LLM accessed through `services.llm` only via `ExplanationService` from graph code**. The module defines `GatewayClient`, a single concrete class backed by the OpenAI-format REST API. Production points at the enterprise LLM gateway; local development points at the local FastAPI shim, which proxies to Ollama's OpenAI-compatible endpoint. The async graph path uses `AsyncOpenAI`; the compatibility sync path uses `OpenAI`. Both are wrapped with `langsmith.wrappers.wrap_openai` so LLM calls appear in LangSmith traces alongside LangGraph node traces — enable with `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.
- **Instrumentation is isolated from graph topology.** Node timing/logging lives in `instrumentation/nodes.py` and supports sync and async callables; run names, LangSmith tags, metadata, and checkpoint `thread_id` config live in `instrumentation/tracing.py`.
- **ExplanationService uses dependency injection for LLM access**. It accepts a `GatewayClient` so tests can mock it at the call site. `main.build_graph()` wires the explanation service into the graph.
- **Monte Carlo RNG is injected via `build_monte_carlo_node(rng=None)`**. Production passes no RNG (a fresh non-seeded generator is created at graph-build time). Tests pass `np.random.default_rng(42)` for determinism. The fixed seed in the previous implementation was development convenience, not policy — same inputs do not guarantee the same `ConfidenceScore` in production.
- **CustomerProfile is mocked** in v1. All data interfaces are designed to be replaced by real API calls without changing node signatures.
- **LP solver: cvxpy**. Constraints are expressed as Python expressions that read like math (readable, auditable).
- **Unit tests on the LP optimizer** are in scope (known inputs → known optimal output). Not over-engineering for a financial product — table stakes.

## Out of Scope

- **Multi-year LP with income growth** — income, expenses, and IRS limits are held fixed; year-over-year changes to these are deferred.
- **Locally implemented probability-of-ruin logic** — `ConfidenceScore` is a retirement-readiness result returned by the Monte Carlo processor; the graph should consume it rather than reimplementing the decumulation calculation locally.
- **Incremental low-confidence reruns** — `PlanRevisionIntake` produces a complete **RevisedPlanScenario** before rerunning optimization; the graph should not rerun after each individual intake answer.
- **Repeated low-confidence reruns** — a **RevisionAttemptLimit** of one automatic revision attempt; further low-confidence outcomes stop with the best available result and next-step guidance.
- **Social Security benefit calculation** — excluded; expected Social Security income may be supplied as part of `ExpectedRetirementIncome`, but the graph does not estimate benefits.
- **Per-account differential returns** — single `ExpectedReturn` applied uniformly.
- **Asset allocation as a decision variable** — fixed glide path / single return assumption.
- **Strategic Roth conversions, backdoor Roth, mega backdoor Roth** — out of scope.
- **State income tax** — only federal marginal rates are modeled.
- **Conversational input gathering** — takes a fully-populated `CustomerProfile`; LLM-driven conversation loop is a future addition.
- **Interactive low-confidence revision workflow** — `PlanRevisionIntake`, `RevisedPlanScenario`, and comparison output are deferred.

## Project Goal

Current goal: build a best-practice LangGraph pipeline with a fan-out/fan-in parallel analysis architecture. Quality bar is idiomatic LangGraph, clean contracts, and testability. The three parallel analysis branches demonstrate reducer-backed concurrent state updates, and every LLM call from the graph goes through `ExplanationService` and the configured gateway.

Future goal: make the graph interactive using LangGraph human-in-the-loop / interrupt patterns — `PlanRevisionIntake`, `RevisedPlanScenario`, and comparison output.

## Next Steps

- **Human-in-the-loop revision workflow** — implement `PlanRevisionIntake` using LangGraph interrupt patterns; present model-informed levers (retirement age, expenses, savings targets) and rerun optimization on the approved `RevisedPlanScenario`.
- **Replace mocked profile with real API** — `src/lang_graph_state/nodes/load_profile.py` returns a hardcoded `CustomerProfile`; wire it to the actual customer data API when available.
- **HSA age-55 catch-up** — the two-phase model uses base HSA limits in both phases; a third phase (age 55+) would correctly capture the HSA catch-up window.
- **Multi-year LP** — income growth, expense growth, and IRS limit indexing are held fixed in v1; a year-by-year LP formulation deferred to v2.

## Flagged Ambiguities

- **Low-confidence workflow escalation** — resolved for v2 as **PlanRevisionIntake** followed by another optimization attempt when the customer changes adjustable planning levers; advisor escalation and required disclosure generation remain separate unresolved workflows.
