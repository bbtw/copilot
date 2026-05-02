# Retirement Plan Optimizer

A LangGraph pipeline that optimizes a customer's retirement savings strategy. Given a customer profile, it determines the optimal annual contribution amounts across account types to maximize **after-tax projected wealth** at retirement.

## What it does

1. **Loads a customer profile** — age, income, expenses, existing balances, employer match, tax rates
2. **Runs a cvxpy LP optimizer** — finds the optimal annual contribution allocation across six account types, in two phases (pre-50 and post-50 401k/IRA catch-up)
3. **Runs a Monte Carlo simulation** — 1,000 paths varying market returns and inflation; reports percentiles in today's dollars and scores retirement readiness
4. **Generates a plain-English explanation** — via Claude (claude-haiku-4-5)
5. **Formats and prints the result** — allocation, projected wealth, percentile distribution, confidence score

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

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
uv run python main.py
```

Example output:

```
============================================================
RETIREMENT PLAN SUMMARY
============================================================

Annual Contributions (Pre-50):
  hsa                       $4,150
  401k                      $7,200    ← captures full employer match
  roth_ira                  $7,000
  taxable_brokerage        $16,650    ← overflow

Annual Contributions (Post-50 / Catch-up):
  hsa                       $4,150
  401k                      $7,200
  roth_ira                  $8,000
  taxable_brokerage        $14,650

Projected After-Tax Wealth at Retirement:   $1,847,000
                                             
Monte Carlo percentiles (today's dollars):
  10th percentile:          $1,102,000
  50th percentile:          $1,534,000
  90th percentile:          $2,198,000
  Confidence score:              42.3%

Explanation:
  ...
============================================================
```

## Testing

Run the unit tests:

```bash
uv run pytest
```

The current test suite covers the LP optimizer constraints and Monte Carlo projection behavior. Tests do not require an Anthropic API key.

## Project structure

```
├── main.py              ← builds and runs the graph
├── state.py             ← RetirementPlanState TypedDict
├── models.py            ← data classes
├── llm_gateway.py       ← thin Claude abstraction
└── nodes/
    ├── load_profile.py  ← mocked CustomerProfile
    ├── lp_optimizer.py   ← cvxpy LP (two-phase, multi-year)
    ├── monte_carlo.py   ← stochastic simulation
    ├── explanation.py   ← LLM narration
    └── format_output.py ← final result assembly
```

## How state flows through the graph

The graph is built as a `StateGraph(RetirementPlanState)`. `RetirementPlanState` is a `TypedDict` that names every key the pipeline may add:

```python
customer_profile
contribution_allocation
projected_wealth
wealth_distribution
confidence_score
confidence_band
explanation
result
```

`main.py` starts the graph with `app.invoke({})`, so the initial state is empty. Each node receives the accumulated state and returns only the keys it owns. LangGraph merges that returned partial dict into the shared state before calling the next node.

The v1 state sequence is:

| Node | Reads | Returns |
|---|---|---|
| `load_customer_profile` | initial empty state | `customer_profile` |
| `run_lp_optimizer` | `customer_profile` | `contribution_allocation`, `projected_wealth` |
| `run_monte_carlo` | `customer_profile`, `contribution_allocation`, `projected_wealth` | `wealth_distribution`, `confidence_score`, `confidence_band` |
| `route_by_confidence_band` | `confidence_band` | low, medium, or high explanation route |
| confidence-specific explanation node | `customer_profile`, `contribution_allocation`, `projected_wealth`, `wealth_distribution`, `confidence_score`, `confidence_band` | `explanation` |
| `format_output` | all prior outputs | `result` |

Nodes do not mutate the state object in place. The contract is: read required keys from the incoming state, return a small dict with newly produced keys, and let LangGraph assemble the final state.

## Key design decisions

- **Two-phase LP** — separate allocation vectors for pre-50 and post-50 years, capturing age-50 catch-up contribution limits for 401k and IRA accounts
- **Multi-year compounding** — LP objective is after-tax terminal wealth compounded to retirement; income and limits held fixed (v1 approximation)
- **Combined employer match cap** — match ceiling applies to 401k + Roth 401k combined, matching real-world plan rules
- **After-tax objective** — pre-tax accounts scaled by `(1 − retirement_tax_rate)` so the optimizer correctly favors Roth accounts when current rates exceed retirement rates
- **MC percentiles in today's dollars** — terminal wealth deflated by simulated inflation for interpretable percentiles; confidence measures whether assets can fund planned retirement expenses after expected retirement income
- **Confidence-based routing** — the graph branches after Monte Carlo by `confidence_band`; route-specific explanation functions use distinct guidance while sharing common prompt assembly

See `CONTEXT.md` for the full domain glossary and architecture decisions.

For a deeper walkthrough of the graph, state flow, and development boundaries, open
`docs/developer-guide.html` in a browser.

## v2 / out of scope

- Multi-year LP with income growth and IRS limit indexing
- Decumulation modeling (withdrawal sequencing, longevity)
- Social Security benefit calculation; expected retirement income is supplied as customer profile data
- Per-account differential returns
- Roth conversions, backdoor Roth
- State income tax
- Conversational input gathering (LLM-driven profile collection)
- Refactor graph nodes into orchestration wrappers that call service-layer implementations, so optimization, simulation, profile loading, and explanation logic do not live directly in node functions
- Confidence-based explanation routing: low, medium, and high confidence projection outcomes should eventually receive different explanation treatment, but thresholds and behaviors are still unresolved
