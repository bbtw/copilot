import logging
from time import perf_counter

from langchain_core.tools import tool

from lang_graph_state.domain.models import CustomerProfile
from lang_graph_state.services.lp_solver import solve_lp

logger = logging.getLogger(__name__)


def make_lp_sensitivity_tools(profile: CustomerProfile) -> list:
    """
    Return LP sensitivity probe tools bound to the given profile.
    Each tool perturbs one input, re-runs the LP, and returns a readable comparison.
    """

    @tool
    def probe_savings_capacity_delta(expense_reduction: float) -> str:
        """
        Check how the LP allocation and projected wealth change if annual expenses decrease
        by expense_reduction dollars (savings capacity increases by that amount).
        """
        return _run_probe(
            profile.model_copy(update={"annual_expenses": profile.annual_expenses - expense_reduction}),
            label=f"${expense_reduction:,.0f} lower annual expenses",
        )

    @tool
    def probe_retirement_age_shift(additional_years: int) -> str:
        """
        Check how the LP result changes if retirement age is pushed out by additional_years years.
        """
        return _run_probe(
            profile.model_copy(update={"retirement_age": profile.retirement_age + additional_years}),
            label=f"retirement at age {profile.retirement_age + additional_years} (+{additional_years} yrs)",
        )

    @tool
    def probe_employer_match_cap(new_cap_rate: float) -> str:
        """
        Check how the LP result changes if the employer match cap rate changes to new_cap_rate
        (e.g. 0.06 for 6% of salary).
        """
        return _run_probe(
            profile.model_copy(update={"employer_match_cap": new_cap_rate}),
            label=f"employer match cap at {new_cap_rate:.1%} of salary",
        )

    return [probe_savings_capacity_delta, probe_retirement_age_shift, probe_employer_match_cap]


def run_default_lp_sensitivity_probes(profile: CustomerProfile) -> list[str]:
    return [
        _run_probe(
            profile.model_copy(update={"annual_expenses": profile.annual_expenses - 5_000}),
            label="$5,000 lower annual expenses",
        ),
        _run_probe(
            profile.model_copy(update={"retirement_age": profile.retirement_age + 2}),
            label=f"retirement at age {profile.retirement_age + 2} (+2 yrs)",
        ),
        _run_probe(
            profile.model_copy(update={"employer_match_cap": profile.employer_match_cap + 0.02}),
            label=f"employer match cap at {profile.employer_match_cap + 0.02:.1%} of salary",
        ),
    ]


def _run_probe(profile: CustomerProfile, *, label: str) -> str:
    logger.info("LP sensitivity probe start label=%s", label)
    started = perf_counter()
    result = solve_lp(profile)
    alloc = result["contribution_allocation"]
    wealth = result["projected_wealth"]
    pre50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.pre50.items() if v > 0)
    post50 = ", ".join(f"{k}=${v:,.0f}" for k, v in alloc.post50.items() if v > 0)
    output = (
        f"Scenario: {label}\n"
        f"Projected after-tax wealth: ${wealth:,.0f}\n"
        f"Pre-50 allocation: {pre50 or 'none'}\n"
        f"Post-50 allocation: {post50 or 'none'}"
    )
    logger.info(
        "LP sensitivity probe finish label=%s elapsed=%.2fs output_chars=%s",
        label,
        perf_counter() - started,
        len(output),
    )
    return output
