# Retirement Planning Optimizer

A chat-driven retirement planning optimizer for consumers. The person whose retirement it is talks to the system in natural language; a Gurobi optimization model produces the plan.

## Language

**User**:
The financial unit whose retirement is being optimized and who operates the chat — one individual or one married couple pooled into a single profile. "Married" surfaces in exactly one place: filing status (MFJ brackets and standard deduction). One set of Accounts, one retirement age, one Savings Capacity, one Spending Need; a spouse's Social Security is just a second Guaranteed Income entry. Consumer-facing — there is no advisor in the loop.
_Avoid_: Client, customer, advisor (imply a professional intermediary); household (implies the members are modeled separately — they aren't)

**Account**:
One of exactly three tax-treatment buckets holding the User's assets: Traditional (pre-tax), Roth, or Taxable brokerage. Real-world account types (401(k), IRA) are collapsed into their bucket — the system never distinguishes them. A single combined annual contribution limit (User-overridable) applies to Traditional + Roth; Taxable is unlimited. Employer match is extra Traditional inflow outside Savings Capacity — an input, never a decision.
_Avoid_: Portfolio, fund (imply asset selection, out of scope); 401(k), IRA (real account types the model deliberately doesn't know about)

**Savings Capacity**:
The fixed annual amount the User can save during the Accumulation Phase. An input, not a decision variable.
_Avoid_: Savings rate, contribution (the latter names where money goes, not how much is available)

**Spending Need**:
The fixed annual amount the User must withdraw during the Decumulation Phase. An input, not a decision variable.
_Avoid_: Budget, expenses, income need

**Contribution Split**:
The optimizer's per-year decision of how Savings Capacity is divided across Accounts, subject to IRS contribution limits.
_Avoid_: Allocation (overloaded with asset allocation, which is a User input)

**Withdrawal Schedule**:
The optimizer's per-year decision of how much to withdraw from each Account to meet Spending Need. Traditional and Roth withdrawals are forbidden before age 59½ (no penalty modeling — if Taxable can't bridge the gap, the plan is infeasible and the chat explains why).
_Avoid_: Drawdown plan, distribution strategy

**Roth Conversion**:
The optimizer's per-year decision to move money Traditional → Roth, taxed as ordinary income in the conversion year. Allowed in any year; valuable in low-bracket years. Does not count toward RMD satisfaction.
_Avoid_: Backdoor Roth (a contribution technique, not this)

**RMD**:
The IRS-required minimum Traditional withdrawal from age 73, proportional to balance per the Uniform Lifetime Table. A hard constraint — it is why unconverted Traditional balances eventually get force-taxed.
_Avoid_: Mandatory distribution, forced withdrawal

**Guaranteed Income**:
Optional User-provided income streams the optimizer doesn't control: Social Security (annual benefit + start age, 85% counted as taxable ordinary income — the convex-safe asymptote of the real phase-in) and pension (100% taxable). Reduces the withdrawals needed for Spending Need and occupies bracket space. Benefit amounts are inputs; the system never estimates them, and claiming age is not a decision variable.
_Avoid_: Fixed income (means bonds), entitlements

**Accumulation Phase**:
The years from today until retirement, during which the Contribution Split is decided.

**Decumulation Phase**:
The years from retirement until the Plan Horizon, during which the Withdrawal Schedule is decided.
_Avoid_: Retirement phase, distribution phase

**Plan Horizon**:
The final year of the plan — the end of the Decumulation Phase.
_Avoid_: End of life, life expectancy (the horizon is a planning input, not a mortality prediction)

**Objective**:
The User's per-run choice of what to maximize: **Wealth at Retirement** (assets at the start of the Decumulation Phase) or **Terminal Wealth** (assets at the Plan Horizon). One lifetime model, two selectable objective functions — never blended or stacked.
_Avoid_: Goal, target (those suggest a threshold to hit, not a quantity to maximize)

**After-Tax Value**:
The measurement unit for both Objectives: Roth at face, Taxable at face (its growth is already taxed via annual tax drag), Traditional at face minus bracket-taxed liquidation. Raw balances are never compared across Accounts.
_Avoid_: Net worth, balance (raw balances overstate Traditional dollars and would let the solver game the objective)

**Tax Drag**:
The Taxable Account's growth is taxed annually at the flat capital-gains rate — it grows at the after-tax return, its withdrawals are tax-free, and no cost basis is tracked. A deliberate convex simplification that slightly over-favors tax-advantaged Accounts.
_Avoid_: Basis tracking, realized gains (concepts the model deliberately doesn't have)

**Expected Return**:
A single real (after-inflation) rate of return, User input with a sensible default, applied identically to all three Accounts (Taxable net of Tax Drag). Per-Account returns are deliberately impossible — they would smuggle asset-location advice out of a model that doesn't understand assets.
_Avoid_: Per-account returns, growth rate assumptions (plural)

**Profile**:
The typed, structured object holding everything the solver needs about the User — ages, balances per Account, gross annual income (needed to price the Traditional deduction against the brackets), Savings Capacity, Spending Need, filing status, Guaranteed Income, optional employer match, Expected Return. Filled conversationally by the LLM, validated in code.
_Avoid_: Intake, form, client profile

**Plan**:
The solver's output for one Profile + one Objective: the year-by-year Contribution Split, Withdrawal Schedule, Roth Conversions, resulting balances, and objective value. The only source of financial numbers in the conversation.
_Avoid_: Recommendation, advice (the system shows an optimal schedule under stated assumptions; it does not advise)

**What-If**:
A follow-up question answered by mutating the Profile and re-solving — never by the LLM extrapolating from a previous Plan. First-class in the chat; it is why chat beats a form.
_Avoid_: Scenario analysis, sensitivity (heavier machinery than a re-solve)

**Real Dollars**:
The model's single unit: today's purchasing power. Bracket thresholds, contribution limits, Savings Capacity, and Spending Need stay constant across years; returns are real (after-inflation). No inflation input exists.
_Avoid_: Nominal dollars, future dollars (never appear anywhere in the system — inputs, model, or chat output)

**Tax Model**:
Progressive federal ordinary-income brackets (with filing status and standard deduction as User inputs), plus a flat capital-gains rate on the Taxable Account. No state tax at v1. Kept convex so the optimization stays a pure LP — tax features that break convexity (e.g., Social Security benefit taxation phase-in) are out of scope until that constraint is consciously dropped.
_Avoid_: Effective rate, flat rate (a flat ordinary-income rate collapses the optimization to a corner solution)

### Evaluation

**Simulated User**:
An LLM that plays the User in an eval conversation, permitted to state only what its Fact Card contains.
_Avoid_: Mock user, synthetic user (suggest canned responses — the Simulated User improvises within its facts)

**Fact Card**:
A scenario's ground-truth User facts — the sole source the Simulated User speaks from, and the expected solve_plan arguments for scoring.
_Avoid_: Fixture, test data (undersell that it is both script and answer key)

**Profile Fidelity**:
The eval dimension asking: do the raw solve_plan arguments match the Fact Card exactly — stated fields verbatim, unstated optional fields strictly absent (an explicitly-passed default is a failure, since it may be hardcoded from the prompt rather than sourced from taxdata).
_Avoid_: Extraction accuracy (fidelity includes not inventing fields, not just extracting stated ones)

**Number Faithfulness**:
The eval dimension asking: does every number in the agent's prose match a source value — the solve_plan result (or, for an infeasible Profile, the solver's infeasibility reason), the Fact Card, an age within the plan, or a taxdata default — either exactly or rounded to the significant digits displayed ("$1.2M" matches 1,203,456; "$1.25M" does not; derived arithmetic such as "$40k/year" from a $320k total always fails).
_Avoid_: Hallucination check (derived arithmetic from real figures also fails, not just invented numbers)

**Scenario**:
One eval unit: a Fact Card conversation driven to the first solve_plan call plus the agent's narration turn, under a hard turn cap — reaching the cap without a solve scores as failure ("no-solve"), not as an error.
_Avoid_: Test case (a Scenario's transcript differs run to run; only the Fact Card is fixed)

**Voided Run**:
A Scenario run discarded because the Simulated User stated a number not on its Fact Card (audited with the same extractor as Number Faithfulness); voided runs are retried and reported separately, never counted as agent failures.
_Avoid_: Flake (voids are attributed to the sim side by construction, not unexplained)

## Relationships

- A **User** operates the chat about their own retirement (no third-party profiles)
- The optimizer decides money movements only: the **Contribution Split** (Accumulation Phase) and the **Withdrawal Schedule** (Decumulation Phase)
- **Savings Capacity**, **Spending Need**, and the single **Expected Return** are User inputs — never decision variables
- Taxes are the friction that makes the optimization non-trivial; without them the problem is degenerate
- A run maximizes exactly one **Objective** over one shared lifetime model (Accumulation + Decumulation); under **Wealth at Retirement**, decumulation years must still be feasible (Spending Need met) but don't affect the objective
- A run consumes one **Profile** + one **Objective** and produces one **Plan**; a **What-If** is just another run
- If a Profile is infeasible (e.g. Spending Need can't be met, or Taxable can't bridge to age 59½), there is no Plan — the chat explains the infeasibility instead
- A **Scenario** pairs one **Fact Card** with one **Simulated User**; repeated runs (three per suite invocation) yield two suite scores — the **Profile Fidelity** rate and the **Number Faithfulness** rate — never blended into one number
- A **Voided Run** is charged to the Simulated User, never to the agent; a no-solve run is charged to the agent, never to the harness

## Example dialogue

> **Dev:** "The User asked 'should I be doing Roth conversions?' — can the chat just say yes?"
> **Domain expert:** "The LLM never answers a money question from its own head. It runs a solve. If the optimal Plan shows Roth Conversions of $40k/year from 62 to 69, the chat says that, citing the Plan's numbers. If the Plan shows zero conversions, the honest answer is 'not under your assumptions.'"
>
> **Dev:** "The User has $1M in Traditional and $880k would-be in Roth — which Plan wins Wealth at Retirement?"
> **Domain expert:** "Neither number as stated — the Objective is measured in After-Tax Value. The Traditional million is worth face minus bracket-taxed liquidation; the Roth is face. Raw balances are never compared."
>
> **Dev:** "In an eval run the agent passed expected_return=0.05 explicitly, and 0.05 is the default — Profile Fidelity pass?"
> **Domain expert:** "Fail. The Fact Card never stated a return, so the field must be absent. An agent that volunteers today's default has probably memorized it from the prompt — and today's default isn't tomorrow's."

## Flagged ambiguities

- This repo previously held an advisor-facing "Financial Planning Co-Pilot" with its own glossary — that domain is dead; none of its terms (Advisor, Client, Brief, Wellness Model) carry over.
