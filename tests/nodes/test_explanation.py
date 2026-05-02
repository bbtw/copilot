from models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    WealthDistribution,
)
from nodes import explanation


class FakeExplanationService:
    def __init__(self) -> None:
        self.request = None

    def generate(self, request):
        self.request = request
        return "explanation"


def _state(confidence_band: str) -> dict:
    return {
        "customer_profile": CustomerProfile(
            age=42,
            retirement_age=65,
            annual_income=120_000,
            annual_expenses=85_000,
            retirement_annual_expenses=75_000,
            retirement_years_to_plan=30,
            expected_retirement_income=30_000,
            balances={a: 0.0 for a in ACCOUNT_TYPES},
            employer_match_rate=0.50,
            employer_match_cap=0.06,
            hdhp_enrolled=True,
            current_marginal_tax_rate=0.22,
            assumed_retirement_marginal_tax_rate=0.12,
            filing_status="single",
        ),
        "contribution_allocation": ContributionAllocation(
            pre50={a: 0.0 for a in ACCOUNT_TYPES},
            post50={a: 0.0 for a in ACCOUNT_TYPES},
        ),
        "projected_wealth": 1_000_000,
        "wealth_distribution": WealthDistribution(p10=600_000, p50=900_000, p90=1_300_000),
        "confidence_score": 0.35,
        "confidence_band": confidence_band,
    }


def test_build_explanation_node_adapts_state_to_service_request():
    service = FakeExplanationService()
    node = explanation.build_explanation_node("low", service)

    result = node(_state("low"))

    assert result == {"explanation": "explanation"}
    assert service.request.confidence_band == "low"
    assert service.request.projected_wealth == 1_000_000
    assert service.request.customer_profile.age == 42


def test_explanation_route_mismatch_fails_fast():
    service = FakeExplanationService()
    node = explanation.build_explanation_node("low", service)

    try:
        node(_state("high"))
    except ValueError as exc:
        assert "Explanation route mismatch" in str(exc)
    else:
        raise AssertionError("Expected route mismatch to raise ValueError")
