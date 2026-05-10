from typing import Any

import numpy as np

from lang_graph_state.domain.models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    WealthDistribution,
    classify_confidence,
)
from lang_graph_state.instrumentation.timing import timed

N_PATHS = 1_000
RETURN_MEAN = 0.07
RETURN_STD = 0.15
INFLATION_MEAN = 0.03
INFLATION_STD = 0.01


def _after_tax_multiplier(acct: str, retirement_tax_rate: float) -> float:
    if acct in ("roth_401k", "roth_ira", "hsa"):
        return 1.0
    return 1.0 - retirement_tax_rate


@timed(
    "mc_simulate",
    finish_attrs=lambda result: {
        "p10": f"{result['wealth_distribution'].p10:.2f}",
        "p50": f"{result['wealth_distribution'].p50:.2f}",
        "p90": f"{result['wealth_distribution'].p90:.2f}",
        "confidence": f"{result['confidence_score']:.4f}",
        "band": result["confidence_band"],
    },
)
def simulate(
    profile: CustomerProfile,
    allocation: ContributionAllocation,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Run N_PATHS Monte Carlo retirement simulations.
    Returns wealth_distribution (today's dollars), confidence_score, and confidence_band.
    """
    p = profile
    tax_out = p.assumed_retirement_marginal_tax_rate
    accumulation_years = p.years_to_retirement
    retirement_years = p.retirement_years_to_plan
    total_years = accumulation_years + retirement_years
    multipliers = {a: _after_tax_multiplier(a, tax_out) for a in ACCOUNT_TYPES}

    terminal_wealth_today = np.zeros(N_PATHS)
    retirement_success = np.zeros(N_PATHS, dtype=bool)
    returns = rng.normal(RETURN_MEAN, RETURN_STD, size=(N_PATHS, total_years))
    inflations = rng.normal(INFLATION_MEAN, INFLATION_STD, size=(N_PATHS, total_years))

    for path in range(N_PATHS):
        balances = {a: p.balances.get(a, 0.0) for a in ACCOUNT_TYPES}
        cumulative_inflation = 1.0

        for yr in range(accumulation_years):
            contrib = allocation.for_age(p.age + yr)
            ret = returns[path, yr]
            matched = p.employer_match_rate * min(
                contrib.get("401k", 0.0) + contrib.get("roth_401k", 0.0),
                p.employer_match_cap * p.annual_income,
            )
            for acct in ACCOUNT_TYPES:
                c = contrib.get(acct, 0.0) + (matched if acct == "401k" else 0.0)
                balances[acct] = balances[acct] * (1 + ret) + c
            cumulative_inflation *= (1 + inflations[path, yr])

        after_tax = sum(balances[a] * multipliers[a] for a in ACCOUNT_TYPES)
        terminal_wealth_today[path] = after_tax / cumulative_inflation

        retirement_assets = after_tax
        net_expenses = max(0.0, p.retirement_annual_expenses - p.expected_retirement_income)
        survived = True

        for yr in range(accumulation_years, total_years):
            cumulative_inflation *= (1 + inflations[path, yr])
            retirement_assets = retirement_assets * (1 + returns[path, yr]) - net_expenses * cumulative_inflation
            if retirement_assets < 0:
                survived = False
                break

        retirement_success[path] = survived

    p10, p50, p90 = np.percentile(terminal_wealth_today, [10, 50, 90])
    confidence = float(np.mean(retirement_success))
    confidence_band = classify_confidence(confidence)

    return {
        "wealth_distribution": WealthDistribution(p10=float(p10), p50=float(p50), p90=float(p90)),
        "confidence_score": confidence,
        "confidence_band": confidence_band,
    }
