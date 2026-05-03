from collections.abc import Callable
from typing import Any

import numpy as np
from models import WealthDistribution, ACCOUNT_TYPES, ConfidenceBand
from state import RetirementPlanState

N_PATHS = 1_000
RETURN_MEAN = 0.07
RETURN_STD = 0.15
INFLATION_MEAN = 0.03
INFLATION_STD = 0.01


def classify_confidence(score: float) -> ConfidenceBand:
    if score < 0.70:
        return "low"
    if score < 0.85:
        return "medium"
    return "high"


def after_tax_multiplier(acct: str, retirement_tax_rate: float) -> float:
    if acct in ("roth_401k", "roth_ira", "hsa"):
        return 1.0
    elif acct in ("401k", "traditional_ira"):
        return 1.0 - retirement_tax_rate
    else:
        return 1.0 - retirement_tax_rate


def build_monte_carlo_node(
        rng: np.random.Generator | None = None,
) -> Callable[[RetirementPlanState], dict[str, Any]]:
    _rng = rng if rng is not None else np.random.default_rng()

    def run_monte_carlo(state: RetirementPlanState) -> dict[str, Any]:
        return _run_monte_carlo(state, rng=_rng)

    return run_monte_carlo


def _run_monte_carlo(
    state: RetirementPlanState,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    p = state.customer_profile
    alloc = state.contribution_allocation
    tax_out = p.assumed_retirement_marginal_tax_rate

    accumulation_years = p.years_to_retirement
    retirement_years = p.retirement_years_to_plan
    total_years = accumulation_years + retirement_years
    multipliers = {a: after_tax_multiplier(a, tax_out) for a in ACCOUNT_TYPES}

    terminal_wealth_today = np.zeros(N_PATHS)
    terminal_wealth_nominal = np.zeros(N_PATHS)
    retirement_success = np.zeros(N_PATHS, dtype=bool)

    returns = rng.normal(RETURN_MEAN, RETURN_STD, size=(N_PATHS, total_years))
    inflations = rng.normal(INFLATION_MEAN, INFLATION_STD, size=(N_PATHS, total_years))

    for path in range(N_PATHS):
        balances = {a: p.balances.get(a, 0.0) for a in ACCOUNT_TYPES}
        cumulative_inflation = 1.0

        # Phase 1: Accumulation
        # Simulate annual compounding, adding contributions and matching employer funds.
        for yr in range(accumulation_years):
            current_age = p.age + yr
            contrib = alloc.for_age(current_age)
            ret = returns[path, yr]
            inf = inflations[path, yr]

            # Calculate employer match: rate * eligible contribution up to a maximum cap
            matched_dollars = p.employer_match_rate * min(
                contrib.get("401k", 0.0) + contrib.get("roth_401k", 0.0),
                p.employer_match_cap * p.annual_income,
            )

            for acct in ACCOUNT_TYPES:
                annual_contribution = contrib.get(acct, 0.0)
                if acct == "401k":
                    annual_contribution += matched_dollars
                # Compound the existing balance and add this year's contribution
                balances[acct] = balances[acct] * (1 + ret) + annual_contribution

            cumulative_inflation *= (1 + inf)

        after_tax = sum(
            balances[a] * multipliers[a] for a in ACCOUNT_TYPES
        )
        terminal_wealth_nominal[path] = after_tax
        terminal_wealth_today[path] = after_tax / cumulative_inflation

        retirement_assets = after_tax
        # Calculate the shortfall between expenses and fixed income (e.g., Social Security)
        net_retirement_expenses = max(
            0.0,
            p.retirement_annual_expenses - p.expected_retirement_income,
        )
        survived = True

        # Phase 2: Drawdown (Retirement)
        # Withdraw inflation-adjusted expenses each year and check if we run out of money
        for yr in range(accumulation_years, total_years):
            ret = returns[path, yr]
            inf = inflations[path, yr]

            cumulative_inflation *= (1 + inf)
            retirement_assets *= (1 + ret)
            # Deduct living expenses after adjusting for cumulative inflation
            retirement_assets -= net_retirement_expenses * cumulative_inflation

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
