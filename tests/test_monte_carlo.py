import pytest

from models import ACCOUNT_TYPES, ContributionAllocation, CustomerProfile
import nodes.monte_carlo as monte_carlo


def _profile(**overrides) -> CustomerProfile:
    defaults = dict(
        age=64,
        retirement_age=65,
        annual_income=100_000,
        annual_expenses=90_000,
        balances={a: 0.0 for a in ACCOUNT_TYPES},
        employer_match_rate=0.50,
        employer_match_cap=0.06,
        hdhp_enrolled=True,
        current_marginal_tax_rate=0.22,
        assumed_retirement_marginal_tax_rate=0.0,
        filing_status="single",
    )
    defaults.update(overrides)
    return CustomerProfile(**defaults)


def test_monte_carlo_includes_employer_match_and_compares_nominal_confidence(monkeypatch):
    monkeypatch.setattr(monte_carlo, "N_PATHS", 5)
    monkeypatch.setattr(monte_carlo, "RETURN_MEAN", 0.0)
    monkeypatch.setattr(monte_carlo, "RETURN_STD", 0.0)
    monkeypatch.setattr(monte_carlo, "INFLATION_MEAN", 0.0)
    monkeypatch.setattr(monte_carlo, "INFLATION_STD", 0.0)

    alloc = ContributionAllocation(
        pre50={a: 0.0 for a in ACCOUNT_TYPES},
        post50={a: 0.0 for a in ACCOUNT_TYPES},
    )
    alloc.post50["401k"] = 6_000

    result = monte_carlo.run_monte_carlo(
        {
            "customer_profile": _profile(),
            "contribution_allocation": alloc,
            "projected_wealth": 9_000,
        }
    )

    assert result["wealth_distribution"].p50 == pytest.approx(9_000)
    assert result["confidence_score"] == pytest.approx(1.0)
