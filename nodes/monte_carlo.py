import numpy as np
from models import WealthDistribution, ACCOUNT_TYPES, ConfidenceBand
from state import RetirementPlanState

N_PATHS = 1_000
RETURN_MEAN = 0.07
RETURN_STD = 0.15
INFLATION_MEAN = 0.03
INFLATION_STD = 0.01

RNG = np.random.default_rng(seed=42)


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


def run_monte_carlo(state: RetirementPlanState) -> dict:
    p = state["customer_profile"]
    alloc = state["contribution_allocation"]
    tax_out = p.assumed_retirement_marginal_tax_rate

    accumulation_years = p.years_to_retirement
    retirement_years = p.retirement_years_to_plan
    total_years = accumulation_years + retirement_years
    multipliers = {a: after_tax_multiplier(a, tax_out) for a in ACCOUNT_TYPES}

    terminal_wealth_today = np.zeros(N_PATHS)
    terminal_wealth_nominal = np.zeros(N_PATHS)
    retirement_success = np.zeros(N_PATHS, dtype=bool)

    returns = RNG.normal(RETURN_MEAN, RETURN_STD, size=(N_PATHS, total_years))
    inflations = RNG.normal(INFLATION_MEAN, INFLATION_STD, size=(N_PATHS, total_years))

    for path in range(N_PATHS):
        balances = {a: p.balances.get(a, 0.0) for a in ACCOUNT_TYPES}
        cumulative_inflation = 1.0

        for yr in range(accumulation_years):
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

        retirement_assets = after_tax
        net_retirement_expenses = max(
            0.0,
            p.retirement_annual_expenses - p.expected_retirement_income,
        )
        survived = True

        for yr in range(accumulation_years, total_years):
            ret = returns[path, yr]
            inf = inflations[path, yr]
            cumulative_inflation *= (1 + inf)
            retirement_assets *= (1 + ret)
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
