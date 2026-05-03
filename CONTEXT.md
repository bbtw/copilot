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
An internal explanation-routing category returned by `MonteCarloProjection` and derived from `ConfidenceScore`; canonical values are `low`, `medium`, and `high`. The band may shape graph routing and prompt strategy, but client-facing explanations should translate it into softer projection-risk language rather than presenting the raw band label as a plan quality judgment.
_Avoid_: Customer-facing headline labels like "low confidence plan"

Confidence-specific explanation routes are part of the graph story. Route-specific node wrappers must fail fast if the routed confidence policy disagrees with the `ConfidenceBand` in state; this indicates graph wiring or state mutation is wrong, not a recoverable customer scenario.

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

**GatewayClient**
A single concrete class in `services/llm.py` that wraps an OpenAI-format REST API call. Configured via environment variables (base URL, model provider, model ID). Points at the enterprise LLM gateway in production and at Ollama locally — the switch is config-level, not code-level. The OpenAI client is wrapped with `langsmith.wrappers.wrap_openai` at construction time so every LLM call is traced in LangSmith when `LANGCHAIN_TRACING_V2=true`; the wrapper is a no-op when tracing is not configured. Injected into `ExplanationService`; tests mock it at the call site.
_Avoid_: Provider-specific SDK imports, multiple adapter classes, Protocol or structural-typing indirection, switching logic in application code

**RetirementPlanResult**
The final pipeline output: `ContributionAllocation`, `ProjectedWealth`, `WealthDistribution`, `ConfidenceScore`, `ConfidenceBand`, and a plain-English explanation generated by the LLM. Returned as a structured dict with a print helper for human-readable summary.

**RetirementPlanState**
The accumulated LangGraph state passed through the retirement planning pipeline, containing each node's completed outputs.

## Pipeline (v1)

```
load_customer_profile
       ↓
run_lp_optimizer        ← maximizes after-tax ProjectedWealth; outputs ContributionAllocation
       ↓
run_monte_carlo         ← outputs WealthDistribution (today's dollars) + ConfidenceScore
       ↓
route_by_confidence_band
       ↓
generate_low_confidence_explanation | generate_medium_confidence_explanation | generate_high_confidence_explanation
                         ← LLM narrates the plan in plain English
       ↓
format_output           ← structured dict + human-readable summary
```

## Project Structure

```
lang-graph-state/
├── main.py                  ← builds and runs the LangGraph graph
├── state.py                 ← RetirementPlanState Pydantic model
├── models.py                ← CustomerProfile, ContributionAllocation, WealthDistribution, RetirementPlanResult
├── docs/
│   └── developer-guide.html ← new-developer teaching guide for graph state flow and boundaries
├── services/
│   ├── explanation.py       ← explanation prompt assembly and LLM orchestration
│   └── llm.py               ← GatewayClient wrapping OpenAI-format REST API (Ollama locally, enterprise gateway in prod)
└── nodes/
    ├── load_profile.py      ← mocked CustomerProfile (42yo, $120k income, HDHP enrolled)
    ├── lp_optimizer.py      ← cvxpy LP; sets contribution_allocation + projected_wealth
    ├── monte_carlo.py       ← 1,000-path simulation; sets wealth_distribution + confidence_score + confidence_band
    ├── explanation.py       ← thin confidence-specific explanation route wrappers
    └── format_output.py     ← assembles RetirementPlanResult
```

Each node is a plain function `(state: RetirementPlanState) -> dict[str, Any]`. The graph starts with `app.invoke({})`, so the initial state is empty. Every node reads the attributes it requires from the accumulated Pydantic state object and returns only the keys it produces; LangGraph merges that partial dict into the state before invoking the next node.

| Node | Reads | Returns |
|---|---|---|
| `load_customer_profile` | initial empty state | `customer_profile` |
| `run_lp_optimizer` | `customer_profile` | `contribution_allocation`, `projected_wealth` |
| `run_monte_carlo` | `customer_profile`, `contribution_allocation`, `projected_wealth` | `wealth_distribution`, `confidence_score`, `confidence_band` |
| `route_by_confidence_band` | `confidence_band` | one of `generate_low_confidence_explanation`, `generate_medium_confidence_explanation`, or `generate_high_confidence_explanation` |
| confidence-specific explanation node | `customer_profile`, `contribution_allocation`, `projected_wealth`, `wealth_distribution`, `confidence_score`, `confidence_band` | `explanation` |
| `format_output` | all prior outputs | `result` |

## Architecture Decisions

- **LP → MC is sequential** (not iterative) in v1. The LP maximizes wealth deterministically; MC scores the resulting plan stochastically. Iterative feedback loop deferred to v2.
- **LP and MC are production black boxes**. This repo may simulate both implementations locally, but production will call separate LP optimizer and Monte Carlo APIs. Keep the shared seam at the input details and result contracts; do not factor LP and MC calculation rules into shared implementation code.
- **LP remains an accumulation optimizer in v2**. Even after `ConfidenceScore` becomes a retirement-readiness probability, the LP objective remains after-tax `ProjectedWealth`; Monte Carlo owns readiness scoring and decumulation-style survival analysis.
- **LP is multi-year, two-phase**. Solves for two allocation vectors: pre-50 and post-50, reflecting the IRS catch-up contribution increase at age 50. Account balances compound across both phases; income, expenses, and tax rates are held fixed. Income growth and IRS limit indexing are deferred to v2. For customers already 50+, the pre-50 phase is skipped.
- **Objective is after-tax wealth**, using `assumed_retirement_marginal_tax_rate` to scale pre-tax balances. Without this, the LP would always favor Traditional accounts (wrong answer).
- **Asset allocation is fixed** — a single `ExpectedReturn` applied uniformly. Per-account differential returns deferred to v2.
- **Employer match is modeled as a subsidy** in the objective (piecewise-linear `min`), not a constraint.
- **Roth IRA phase-outs are input preprocessing**, not LP constraints — collapsed to fixed limits before the LP runs. Traditional IRA deductibility phase-outs are deferred.
- **MC reports distribution percentiles in today's dollars** by deflating with simulated inflation. Confidence is returned by the Monte Carlo processor as the probability that retirement income and assets cover `RetirementAnnualExpenses` for `RetirementYearsToPlan` years after the customer's retirement age.
- **LLM accessed through `services.llm`**. The module defines `GatewayClient`, a single concrete class backed by the OpenAI-format REST API. Production points at the enterprise LLM gateway; local development points at Ollama. The switch is config-level (env vars), not code-level. The client is wrapped with `langsmith.wrappers.wrap_openai` so LLM calls appear in LangSmith traces alongside LangGraph node traces — enable with `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`.
- **ExplanationService uses dependency injection for LLM access**. It accepts a `GatewayClient` so tests can mock it at the call site. `main.build_graph()` wires the explanation service into the graph.
- **Explanation nodes set the service-wrapper pattern**. Graph construction wires an `ExplanationService`; confidence-specific explanation nodes remain visible in the graph but only adapt `RetirementPlanState` to `PlanExplanationRequest` and return the `explanation` state patch.
- **Monte Carlo RNG is injected via `build_monte_carlo_node(rng=None)`**. Production passes no RNG (a fresh non-seeded generator is created at graph-build time). Tests pass `np.random.default_rng(42)` for determinism. The fixed seed in the previous implementation was development convenience, not policy — same inputs do not guarantee the same `ConfidenceScore` in production.
- **CustomerProfile is mocked** in v1. All data interfaces are designed to be replaced by real API calls without changing node signatures.
- **LP solver: cvxpy**. Constraints are expressed as Python expressions that read like math (readable, auditable).
- **Unit tests on the LP optimizer** are in scope (known inputs → known optimal output). Not over-engineering for a financial product — table stakes.

## Out of Scope (v1)

- **Multi-year LP with income growth** — income, expenses, and IRS limits are held fixed; year-over-year changes to these are deferred to v2.
- **Decumulation modeling** — withdrawal sequencing rules, retirement expenses, longevity uncertainty.
- **Locally implemented probability-of-ruin logic** — `ConfidenceScore` is a retirement-readiness result returned by the Monte Carlo processor; the graph should consume it rather than reimplementing the decumulation calculation locally.
- **Incremental low-confidence reruns** — v2 `PlanRevisionIntake` produces a complete **RevisedPlanScenario** before rerunning optimization; the graph should not rerun after each individual intake answer.
- **Repeated low-confidence reruns** — v2 uses a **RevisionAttemptLimit** of one automatic revision attempt; further low-confidence outcomes stop with the best available result and next-step guidance.
- **Social Security benefit calculation** — excluded; expected Social Security income may be supplied as part of `ExpectedRetirementIncome`, but the graph does not estimate benefits.
- **Per-account differential returns** — single `ExpectedReturn` applied uniformly.
- **Asset allocation as a decision variable** — fixed glide path / single return assumption.
- **Strategic Roth conversions, backdoor Roth, mega backdoor Roth** — out of scope.
- **State income tax** — only federal marginal rates are modeled.
- **Conversational input gathering** — v1 takes a fully-populated `CustomerProfile`; v2 adds the LLM-driven conversation loop.
- **Remaining service-layer node refactors** — explanation has the target node-wrapper/service shape; optimization, simulation, profile loading, and formatting still contain implementation logic directly.
- **Interactive low-confidence revision workflow** — deferred; the first readiness implementation updates Monte Carlo scoring and explanation routing, but does not yet implement `PlanRevisionIntake`, `RevisedPlanScenario`, or comparison output.

## Project Goal

v1 goal: build a best-practice LangGraph single-pass pipeline. Quality bar is idiomatic LangGraph, clean contracts, and testability — not feature completeness.

v2 goal: make the graph interactive using LangGraph human-in-the-loop / interrupt patterns. The graph pauses after a low-confidence result, presents candidate levers via `PlanRevisionIntake`, waits for customer approval of a `RevisedPlanScenario`, then reruns the LP optimizer and Monte Carlo with the revised inputs. v2 is not in scope until v1 is solid.

## Flagged Ambiguities

- **Low-confidence workflow escalation** — resolved for v2 as **PlanRevisionIntake** followed by another optimization attempt when the customer changes adjustable planning levers; advisor escalation and required disclosure generation remain separate unresolved workflows.
