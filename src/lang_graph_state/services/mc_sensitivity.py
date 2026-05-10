import numpy as np
from langchain_core.tools import tool

from lang_graph_state.domain.models import ContributionAllocation, CustomerProfile
from lang_graph_state.instrumentation.timing import timed
from lang_graph_state.services.mc_simulator import simulate

_PROBE_RNG_SEED = 42


def make_mc_sensitivity_tools(profile: CustomerProfile, allocation: ContributionAllocation) -> list:
    """
    Return MC sensitivity probe tools bound to the given profile and allocation.
    Probes use a fixed RNG seed so results are comparable across calls.
    """

    @tool
    def probe_retirement_expense_delta(annual_expense_delta: float) -> str:
        """
        Check how confidence score and wealth distribution change if retirement annual expenses
        shift by annual_expense_delta dollars (positive = higher, negative = lower).
        """
        return _run_probe(
            profile.model_copy(update={
                "retirement_annual_expenses": profile.retirement_annual_expenses + annual_expense_delta
            }),
            allocation,
            label=f"${annual_expense_delta:+,.0f}/yr retirement expenses",
        )

    @tool
    def probe_retirement_duration_extension(additional_years: int) -> str:
        """
        Check how confidence score changes if the retirement planning horizon extends
        by additional_years years.
        """
        return _run_probe(
            profile.model_copy(update={
                "retirement_years_to_plan": profile.retirement_years_to_plan + additional_years
            }),
            allocation,
            label=f"{profile.retirement_years_to_plan + additional_years} retirement years (+{additional_years})",
        )

    @tool
    def probe_net_withdrawal_rate(annual_portfolio_withdrawal: float) -> str:
        """
        Check confidence score if the net annual portfolio withdrawal is set to
        annual_portfolio_withdrawal dollars (overrides net expenses for this probe only).
        """
        return _run_probe(
            profile.model_copy(update={
                "retirement_annual_expenses": annual_portfolio_withdrawal + profile.expected_retirement_income
            }),
            allocation,
            label=f"${annual_portfolio_withdrawal:,.0f}/yr net portfolio withdrawal",
        )

    return [probe_retirement_expense_delta, probe_retirement_duration_extension, probe_net_withdrawal_rate]


def run_default_mc_sensitivity_probes(profile: CustomerProfile, allocation: ContributionAllocation) -> list[str]:
    net_withdrawal = max(0.0, profile.retirement_annual_expenses - profile.expected_retirement_income)
    return [
        _run_probe(
            profile.model_copy(update={"retirement_annual_expenses": profile.retirement_annual_expenses + 5_000}),
            allocation,
            label="$+5,000/yr retirement expenses",
        ),
        _run_probe(
            profile.model_copy(update={"retirement_years_to_plan": profile.retirement_years_to_plan + 5}),
            allocation,
            label=f"{profile.retirement_years_to_plan + 5} retirement years (+5)",
        ),
        _run_probe(
            profile.model_copy(update={"retirement_annual_expenses": net_withdrawal * 0.9 + profile.expected_retirement_income}),
            allocation,
            label=f"${net_withdrawal * 0.9:,.0f}/yr net portfolio withdrawal",
        ),
    ]


@timed("mc_sensitivity_probe", finish_attrs=lambda output: {"output_chars": len(output)})
def _run_probe(profile: CustomerProfile, allocation: ContributionAllocation, *, label: str) -> str:
    result = simulate(profile, allocation, rng=np.random.default_rng(_PROBE_RNG_SEED))
    dist = result["wealth_distribution"]
    score = result["confidence_score"]
    band = result["confidence_band"]
    return (
        f"Scenario: {label}\n"
        f"Confidence score: {score:.1%} ({band})\n"
        f"Wealth p10/p50/p90 (today's $): ${dist.p10:,.0f} / ${dist.p50:,.0f} / ${dist.p90:,.0f}"
    )
