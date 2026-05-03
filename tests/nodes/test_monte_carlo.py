import numpy as np
import pytest

from models import ACCOUNT_TYPES, ContributionAllocation, CustomerProfile
import nodes.monte_carlo as monte_carlo
from nodes.monte_carlo import build_monte_carlo_node
from state import RetirementPlanState


def _profile(**overrides) -> CustomerProfile:
    defaults = dict(
        age=64,
        retirement_age=65,
        annual_income=100_000,
        annual_expenses=90_000,
        retirement_annual_expenses=9_000,
        retirement_years_to_plan=1,
        expected_retirement_income=0,
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


def test_monte_carlo_includes_employer_match_and_scores_retirement_survival(monkeypatch):
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

    run = build_monte_carlo_node(rng=np.random.default_rng(42))
    result = run(
        RetirementPlanState(
            customer_profile=_profile(),
            contribution_allocation=alloc,
            projected_wealth=9_000,
        )
    )

    assert result["wealth_distribution"].p50 == pytest.approx(9_000)
    assert result["confidence_score"] == pytest.approx(1.0)
    assert result["confidence_band"] == "high"


def test_expected_retirement_income_offsets_retirement_expenses(monkeypatch):
    monkeypatch.setattr(monte_carlo, "N_PATHS", 5)
    monkeypatch.setattr(monte_carlo, "RETURN_MEAN", 0.0)
    monkeypatch.setattr(monte_carlo, "RETURN_STD", 0.0)
    monkeypatch.setattr(monte_carlo, "INFLATION_MEAN", 0.0)
    monkeypatch.setattr(monte_carlo, "INFLATION_STD", 0.0)

    alloc = ContributionAllocation(
        pre50={a: 0.0 for a in ACCOUNT_TYPES},
        post50={a: 0.0 for a in ACCOUNT_TYPES},
    )

    state = RetirementPlanState(
        customer_profile=_profile(
            balances={"401k": 5_000, **{a: 0.0 for a in ACCOUNT_TYPES if a != "401k"}},
            retirement_annual_expenses=30_000,
            expected_retirement_income=25_000,
        ),
        contribution_allocation=alloc,
        projected_wealth=1_000_000,
    )

    run = build_monte_carlo_node(rng=np.random.default_rng(42))
    result = run(state)

    assert result["wealth_distribution"].p50 == pytest.approx(5_000)
    assert result["confidence_score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.399, "low"),
        (0.699, "low"),
        (0.70, "medium"),
        (0.849, "medium"),
        (0.85, "high"),
    ],
)
def test_classify_confidence_uses_documented_thresholds(score, band):
    assert monte_carlo.classify_confidence(score) == band
