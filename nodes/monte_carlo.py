import numpy as np
from models import WealthDistribution, ACCOUNT_TYPES
from state import RetirementPlanState

N_PATHS = 1_000
RETURN_MEAN = 0.07
RETURN_STD = 0.15
INFLATION_MEAN = 0.03
INFLATION_STD = 0.01

RNG = np.random.default_rng(seed=42)


def after_tax_multiplier(acct: str, retirement_tax_rate: float) -> float:
    if acct in ("roth_401k", "roth_ira", "hsa"):
        return 1.0
    elif acct in ("401k", "traditional_ira"):
        return 1.0 - retirement_tax_rate
    else:
        return 1.0 - retirement_tax_rate


def run_monte_carlo(state: RetirementPlanState) -> dict:
    p = state["customer_profile"]
    alloc = state["contribution_allocation"]
    lp_wealth = state["projected_wealth"]
    tax_out = p.assumed_retirement_marginal_tax_rate

    years = p.years_to_retirement
    multipliers = {a: after_tax_multiplier(a, tax_out) for a in ACCOUNT_TYPES}

    terminal_wealth_today = np.zeros(N_PATHS)
    terminal_wealth_nominal = np.zeros(N_PATHS)

    returns = RNG.normal(RETURN_MEAN, RETURN_STD, size=(N_PATHS, years))
    inflations = RNG.normal(INFLATION_MEAN, INFLATION_STD, size=(N_PATHS, years))

    for path in range(N_PATHS):
        balances = {a: p.balances.get(a, 0.0) for a in ACCOUNT_TYPES}
        cumulative_inflation = 1.0

        for yr in range(years):
            current_age = p.age + yr
            contrib = alloc.for_age(current_age)
            ret = returns[path, yr]
            inf = inflations[path, yr]
            matched_dollars = p.employer_match_rate * min(
                contrib.get("401k", 0.0) + contrib.get("roth_401k", 0.0),
                p.employer_match_cap * p.annual_income,
            )

            for acct in ACCOUNT_TYPES:
                annual_contribution = contrib.get(acct, 0.0)
                if acct == "401k":
                    annual_contribution += matched_dollars
                balances[acct] = balances[acct] * (1 + ret) + annual_contribution

            cumulative_inflation *= (1 + inf)

        after_tax = sum(
            balances[a] * multipliers[a] for a in ACCOUNT_TYPES
        )
        terminal_wealth_nominal[path] = after_tax
        terminal_wealth_today[path] = after_tax / cumulative_inflation

    p10, p50, p90 = np.percentile(terminal_wealth_today, [10, 50, 90])
    confidence = float(np.mean(terminal_wealth_nominal >= lp_wealth))

    return {
        "wealth_distribution": WealthDistribution(p10=float(p10), p50=float(p50), p90=float(p90)),
        "confidence_score": confidence,
    }
