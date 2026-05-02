# Retirement Plan Optimizer — Domain Context

## Purpose
A LangGraph pipeline that optimizes a customer's retirement savings strategy for a large financial organization (Vanguard-scale). Given a customer profile, it determines the optimal annual contribution amounts across account types to maximize **after-tax** projected wealth at retirement.

## Glossary

**CustomerProfile**
All inputs describing a customer's financial situation:
- `age`, `retirement_age`
- `annual_income`, `annual_expenses`
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

**MonteCarloProjection**
A simulation of 1,000+ retirement paths run after the LP produces a `ContributionAllocation`. Varies market returns and inflation stochastically. Includes employer match dollars using the same combined 401(k) + Roth 401(k) cap as the LP. Returns a `WealthDistribution` reported in **today's dollars** (deflated by simulated inflation) and a `ConfidenceScore` computed against nominal terminal wealth.

**Production Simulation**
This repository simulates the shape of the production retirement planning pipeline. In production, the LP optimizer and Monte Carlo engine are separate black-box API calls. The only shared assumption is that both calls receive the same customer and plan input details needed for their calculations. They must not share calculation implementation code.

**WealthDistribution**
The spread of projected after-tax portfolio values at retirement across all Monte Carlo simulation paths (10th / 50th / 90th percentiles).

**ConfidenceScore**
The probability of the customer meeting the LP's nominal `ProjectedWealth` at retirement: `P(mc_nominal_terminal_wealth ≥ lp_projected_wealth)`. The LP's deterministic projection is the target — the MC score answers "how likely is the optimal plan to actually deliver its projected outcome?"

**ConfidenceBand**
A future explainability category derived from `ConfidenceScore`, such as low, medium, or high confidence. Exact thresholds and explanation behavior are unresolved.

**PlanningHorizon**
A fixed retirement age used as the endpoint of the accumulation phase in the Monte Carlo simulations. Decumulation phase (retirement spending, withdrawal sequencing, longevity) is not modeled in v1.

**RetirementPlanResult**
The final pipeline output: `ContributionAllocation`, `ProjectedWealth`, `WealthDistribution`, `ConfidenceScore`, and a plain-English explanation generated by the LLM. Returned as a structured dict with a print helper for human-readable summary.

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
generate_explanation    ← LLM narrates the plan in plain English
       ↓
format_output           ← structured dict + human-readable summary
```

## Project Structure

```
lang-graph-state/
├── main.py                  ← builds and runs the LangGraph graph
├── state.py                 ← RetirementPlanState TypedDict
├── models.py                ← CustomerProfile, ContributionAllocation, WealthDistribution, RetirementPlanResult
├── llm_gateway.py           ← thin Claude abstraction (default: claude-haiku-4-5)
├── docs/
│   └── developer-guide.html ← new-developer teaching guide for graph state flow and boundaries
└── nodes/
    ├── load_profile.py      ← mocked CustomerProfile (42yo, $120k income, HDHP enrolled)
    ├── lp_optimizer.py      ← cvxpy LP; sets contribution_allocation + projected_wealth
    ├── monte_carlo.py       ← 1,000-path simulation; sets wealth_distribution + confidence_score
    ├── explanation.py       ← LLM call; sets explanation
    └── format_output.py     ← assembles RetirementPlanResult
```

Each node is a plain function `(state: RetirementPlanState) -> dict`. The graph starts with `app.invoke({})`, so the initial state is empty. Every node reads the keys it requires from the accumulated state and returns only the keys it produces; LangGraph merges that partial dict into the state before invoking the next node.

| Node | Reads | Returns |
|---|---|---|
| `load_customer_profile` | initial empty state | `customer_profile` |
| `run_lp_optimizer` | `customer_profile` | `contribution_allocation`, `projected_wealth` |
| `run_monte_carlo` | `customer_profile`, `contribution_allocation`, `projected_wealth` | `wealth_distribution`, `confidence_score` |
| `generate_explanation` | `customer_profile`, `contribution_allocation`, `projected_wealth`, `wealth_distribution`, `confidence_score` | `explanation` |
| `format_output` | all prior outputs | `result` |

## Architecture Decisions

- **LP → MC is sequential** (not iterative) in v1. The LP maximizes wealth deterministically; MC scores the resulting plan stochastically. Iterative feedback loop deferred to v2.
- **LP and MC are production black boxes**. This repo may simulate both implementations locally, but production will call separate LP optimizer and Monte Carlo APIs. Keep the shared seam at the input details and result contracts; do not factor LP and MC calculation rules into shared implementation code.
- **LP is multi-year, two-phase**. Solves for two allocation vectors: pre-50 and post-50, reflecting the IRS catch-up contribution increase at age 50. Account balances compound across both phases; income, expenses, and tax rates are held fixed. Income growth and IRS limit indexing are deferred to v2. For customers already 50+, the pre-50 phase is skipped.
- **Objective is after-tax wealth**, using `assumed_retirement_marginal_tax_rate` to scale pre-tax balances. Without this, the LP would always favor Traditional accounts (wrong answer).
- **Asset allocation is fixed** — a single `ExpectedReturn` applied uniformly. Per-account differential returns deferred to v2.
- **Employer match is modeled as a subsidy** in the objective (piecewise-linear `min`), not a constraint.
- **Roth IRA phase-outs are input preprocessing**, not LP constraints — collapsed to fixed limits before the LP runs. Traditional IRA deductibility phase-outs are deferred.
- **MC reports distribution percentiles in today's dollars** by deflating with simulated inflation. Confidence compares nominal MC terminal wealth to the nominal LP target.
- **LLM accessed through a thin gateway module** that abstracts model selection. Default: `claude-haiku-4-5`. Swappable without touching pipeline nodes.
- **CustomerProfile is mocked** in v1. All data interfaces are designed to be replaced by real API calls without changing node signatures.
- **LP solver: cvxpy**. Constraints are expressed as Python expressions that read like math (readable, auditable).
- **Unit tests on the LP optimizer** are in scope (known inputs → known optimal output). Not over-engineering for a financial product — table stakes.

## Out of Scope (v1)

- **Multi-year LP with income growth** — income, expenses, and IRS limits are held fixed; year-over-year changes to these are deferred to v2.
- **Decumulation modeling** — withdrawal sequencing rules, retirement expenses, longevity uncertainty.
- **"Probability of ruin"** — the v1 `ConfidenceScore` measures probability of hitting a wealth target, not surviving spending in retirement.
- **Social Security** — excluded; can be incorporated later as a deterministic income stream during decumulation.
- **Per-account differential returns** — single `ExpectedReturn` applied uniformly.
- **Asset allocation as a decision variable** — fixed glide path / single return assumption.
- **Strategic Roth conversions, backdoor Roth, mega backdoor Roth** — out of scope.
- **State income tax** — only federal marginal rates are modeled.
- **Conversational input gathering** — v1 takes a fully-populated `CustomerProfile`; v2 adds the LLM-driven conversation loop.
- **Service-layer node refactor** — v1 nodes contain implementation logic directly; v2 should make nodes thin orchestration wrappers around services for optimization, simulation, profile loading, and explanation generation.
- **Confidence-based explanation routing** — v2 should route explanation generation based on the projection outcome. Low, medium, and high confidence plans likely need different explanation strategies, but the thresholds and behaviors are not yet defined.

## Flagged Ambiguities

- **ConfidenceBand thresholds** — low, medium, and high confidence are useful explainability categories, but the project has not decided the numerical `ConfidenceScore` ranges for each band.
- **ConfidenceBand behavior** — the explanation should treat low, medium, and high confidence outcomes differently, but the project has not decided whether that means different prompts, different graph routes, different disclosures, recommendations to re-optimize, or escalation to a human advisor.
