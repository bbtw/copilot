import dataclasses

import numpy as np
from langchain_core.tools import tool

from models import ContributionAllocation, CustomerProfile

_PROBE_RNG_SEED = 42


def make_mc_sensitivity_tools(profile: CustomerProfile, allocation: ContributionAllocation) -> list:
    """
    Return MC sensitivity probe tools that capture the given profile and allocation.
    Each tool perturbs one input and re-runs the Monte Carlo, returning a readable result.
    Probes use a fixed RNG seed so results are comparable across calls.
    """

    @tool
    def probe_retirement_expense_delta(annual_expense_delta: float) -> str:
        """
        Check how the confidence score and wealth distribution change if retirement annual expenses
        shift by annual_expense_delta dollars (positive = higher expenses, negative = lower).
        """
        modified = dataclasses.replace(
            profile,
            retirement_annual_expenses=profile.retirement_annual_expenses + annual_expense_delta,
        )
        label = f"${annual_expense_delta:+,.0f}/yr retirement expenses"
        return _run_mc_probe(modified, allocation, label=label)

    @tool
    def probe_retirement_duration_extension(additional_years: int) -> str:
        """
        Check how the confidence score changes if the retirement planning horizon extends
        by additional_years years (i.e., the plan must fund a longer retirement).
        """
        modified = dataclasses.replace(
            profile,
            retirement_years_to_plan=profile.retirement_years_to_plan + additional_years,
        )
        label = f"{modified.retirement_years_to_plan} retirement years (+{additional_years})"
        return _run_mc_probe(modified, allocation, label=label)

    @tool
    def probe_net_withdrawal_rate(annual_portfolio_withdrawal: float) -> str:
        """
        Check confidence score if the net annual portfolio withdrawal (expenses minus fixed income)
        is set to annual_portfolio_withdrawal dollars. This overrides the current net withdrawal
        for this probe only.
        """
        modified = dataclasses.replace(
            profile,
            retirement_annual_expenses=annual_portfolio_withdrawal + profile.expected_retirement_income,
        )
        label = f"${annual_portfolio_withdrawal:,.0f}/yr net portfolio withdrawal"
        return _run_mc_probe(modified, allocation, label=label)

    return [probe_retirement_expense_delta, probe_retirement_duration_extension, probe_net_withdrawal_rate]


def _run_mc_probe(
    profile: CustomerProfile,
    allocation: ContributionAllocation,
    *,
    label: str,
) -> str:
    from nodes.monte_carlo import _run_monte_carlo
    from state import RetirementPlanState

    state = RetirementPlanState(customer_profile=profile, contribution_allocation=allocation)
    result = _run_monte_carlo(state, rng=np.random.default_rng(_PROBE_RNG_SEED))
    dist = result["wealth_distribution"]
    score = result["confidence_score"]
    band = result["confidence_band"]
    return (
        f"Scenario: {label}\n"
        f"Confidence score: {score:.1%} ({band})\n"
        f"Wealth p10/p50/p90 (today's $): ${dist.p10:,.0f} / ${dist.p50:,.0f} / ${dist.p90:,.0f}"
    )
