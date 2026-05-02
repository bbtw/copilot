from llm_gateway import complete
from state import RetirementPlanState

SYSTEM = (
    "You are a certified financial planner explaining a retirement savings plan to a client. "
    "Be clear, warm, and specific. Avoid jargon. Use dollar amounts and percentages where helpful. "
    "Keep the explanation to 3-4 paragraphs."
)


HIGH_CONFIDENCE_GUIDANCE = (
    "Explain the allocation primarily as the optimizer's selected strategy. "
    "Mention uncertainty briefly and emphasize the major drivers such as employer match, "
    "HSA eligibility, tax treatment, and binding constraints."
)

MEDIUM_CONFIDENCE_GUIDANCE = (
    "Explain the allocation and include a clear uncertainty paragraph. "
    "Describe the projection as plausible but sensitive to market and inflation paths, "
    "and say the client should revisit the plan periodically."
)

LOW_CONFIDENCE_GUIDANCE = (
    "Do not present the plan as likely to meet the client's retirement goal. Explain the allocation, "
    "state that simulated paths often exhaust assets before the planned retirement horizon, and suggest "
    "review levers such as increasing savings capacity, changing retirement age, reducing expenses, "
    "or advisor review. Avoid saying the allocation is bad; frame the issue as retirement-readiness risk."
)


def generate_low_confidence_explanation(state: RetirementPlanState) -> dict:
    return _generate_explanation(state, LOW_CONFIDENCE_GUIDANCE)


def generate_medium_confidence_explanation(state: RetirementPlanState) -> dict:
    return _generate_explanation(state, MEDIUM_CONFIDENCE_GUIDANCE)


def generate_high_confidence_explanation(state: RetirementPlanState) -> dict:
    return _generate_explanation(state, HIGH_CONFIDENCE_GUIDANCE)


def _generate_explanation(state: RetirementPlanState, confidence_guidance: str) -> dict:
    p = state["customer_profile"]
    alloc = state["contribution_allocation"]
    wealth = state["projected_wealth"]
    dist = state["wealth_distribution"]
    score = state["confidence_score"]
    band = state["confidence_band"]
    diagnostics = state.get("optimization_diagnostics")

    pre50_lines = "\n".join(
        f"  - {acct}: ${amt:,.0f}/year"
        for acct, amt in alloc.pre50.items() if amt > 0
    )
    post50_lines = "\n".join(
        f"  - {acct}: ${amt:,.0f}/year"
        for acct, amt in alloc.post50.items() if amt > 0
    )
    optimizer_lines = "No optimizer diagnostics were provided."
    if diagnostics:
        binding_constraints = [
            c for c in diagnostics.constraints
            if c.binding and c.category != "variable_lower_bound"
        ]
        zero_variables = [
            v for v in diagnostics.variables
            if v.at_lower_bound and (v.reduced_cost or 0) > 1e-5
        ]
        positive_variables = [
            v for v in diagnostics.variables
            if v.value > 1e-5
        ]

        binding_lines = "\n".join(
            f"  - {c.name}: slack ${c.slack:,.2f}, shadow price {c.dual_value if c.dual_value is not None else 'n/a'}"
            for c in binding_constraints[:8]
        ) or "  - None"
        selected_lines = "\n".join(
            f"  - {v.name}: ${v.value:,.0f}/year, modeled terminal value per $1 now {v.objective_coefficient:,.2f}"
            for v in positive_variables[:12]
        ) or "  - None"
        zero_lines = "\n".join(
            f"  - {v.name}: reduced-cost signal {v.reduced_cost:,.2f}"
            for v in zero_variables[:8]
        ) or "  - None"

        optimizer_lines = f"""
Solver: {diagnostics.solver}
Status: {diagnostics.status}
Selected variables:
{selected_lines}
Binding non-lower-bound constraints:
{binding_lines}
Zero-selected variables with reduced-cost signals:
{zero_lines}
""".strip()

    prompt = f"""
Customer profile:
- Age: {p.age}, retiring at {p.retirement_age}
- Annual income: ${p.annual_income:,.0f}, annual expenses: ${p.annual_expenses:,.0f}
- Retirement annual expenses: ${p.retirement_annual_expenses:,.0f}
- Retirement years to plan: {p.retirement_years_to_plan}
- Expected retirement income: ${p.expected_retirement_income:,.0f}/year
- Savings capacity: ${p.savings_capacity:,.0f}/year
- Employer match: {p.employer_match_rate:.0%} up to {p.employer_match_cap:.0%} of salary
- HDHP enrolled: {p.hdhp_enrolled}
- Current marginal tax rate: {p.current_marginal_tax_rate:.0%}
- Assumed retirement tax rate: {p.assumed_retirement_marginal_tax_rate:.0%}

Recommended annual contributions (before age 50):
{pre50_lines}

Recommended annual contributions (age 50+, with catch-up):
{post50_lines}

Projected after-tax wealth at retirement: ${wealth:,.0f}
Monte Carlo percentiles (today's dollars):
  - 10th percentile: ${dist.p10:,.0f}
  - 50th percentile: ${dist.p50:,.0f}
  - 90th percentile: ${dist.p90:,.0f}
Confidence score (probability of funding planned retirement years): {score:.1%}
Internal confidence band: {band}

Optimizer diagnostics:
{optimizer_lines}

Please explain this retirement plan to the client in plain English.
Use the optimizer diagnostics to explain why the model selected the major contribution choices.
Do not imply guaranteed results. Describe the allocation as selected by the model under the stated assumptions.
Balance benefits with the relevant constraints, assumptions, and uncertainty.
Confidence-band guidance: {confidence_guidance}
Do not present the raw internal confidence band as a customer-facing label.
""".strip()

    explanation = complete(prompt, system=SYSTEM)
    return {"explanation": explanation}
