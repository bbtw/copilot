import dataclasses

import numpy as np
from langchain_core.tools import tool

from models import CustomerProfile


def make_lp_sensitivity_tools(profile: CustomerProfile) -> list:
    """
    Return LP sensitivity probe tools that capture the given profile.
    Each tool perturbs one input and re-runs the LP, returning a readable result.
    """

    @tool
    def probe_savings_capacity_delta(expense_reduction: float) -> str:
        """
        Check how the LP allocation and projected wealth change if annual expenses decrease
        by expense_reduction dollars (i.e., savings capacity increases by that amount).
        """
        modified = dataclasses.replace(profile, annual_expenses=profile.annual_expenses - expense_reduction)
        return _run_lp_probe(modified, label=f"${expense_reduction:,.0f} lower annual expenses")

    @tool
    def probe_retirement_age_shift(additional_years: int) -> str:
        """
        Check how the LP result changes if the retirement age is pushed out by additional_years years.
        More accumulation time increases compounding but delays retirement.
        """
        modified = dataclasses.replace(profile, retirement_age=profile.retirement_age + additional_years)
        return _run_lp_probe(modified, label=f"retirement at age {modified.retirement_age} (+{additional_years} yrs)")

    @tool
    def probe_employer_match_cap(new_cap_rate: float) -> str:
        """
        Check how the LP result changes if the employer match cap rate changes to new_cap_rate
        (expressed as a fraction of salary, e.g. 0.06 for 6%).
        """
        modified = dataclasses.replace(profile, employer_match_cap=new_cap_rate)
        return _run_lp_probe(modified, label=f"employer match cap at {new_cap_rate:.1%} of salary")

    return [probe_savings_capacity_delta, probe_retirement_age_shift, probe_employer_match_cap]


def _run_lp_probe(profile: CustomerProfile, *, label: str) -> str:
    from nodes.lp_optimizer import run_lp_optimizer
    from state import RetirementPlanState

    result = run_lp_optimizer(RetirementPlanState(customer_profile=profile))
    alloc = result["contribution_allocation"]
    wealth = result["projected_wealth"]

    pre50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.pre50.items() if v > 0)
    post50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.post50.items() if v > 0)
    return (
        f"Scenario: {label}\n"
        f"Projected after-tax wealth: ${wealth:,.0f}\n"
        f"Pre-50 allocation: {pre50 or 'none'}\n"
        f"Post-50 allocation: {post50 or 'none'}"
    )
