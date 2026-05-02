from models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    WealthDistribution,
)
from nodes import explanation


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


def test_low_confidence_route_uses_low_confidence_guidance(monkeypatch):
    captured = {}

    def fake_complete(prompt: str, system: str = "") -> str:
        captured["prompt"] = prompt
        return "explanation"

    monkeypatch.setattr(explanation, "complete", fake_complete)

    result = explanation.generate_low_confidence_explanation(_state("low"))

    assert result == {"explanation": "explanation"}
    assert "Do not present the plan as likely to meet the client's retirement goal" in captured["prompt"]


def test_high_confidence_route_uses_high_confidence_guidance(monkeypatch):
    captured = {}

    def fake_complete(prompt: str, system: str = "") -> str:
        captured["prompt"] = prompt
        return "explanation"

    monkeypatch.setattr(explanation, "complete", fake_complete)

    result = explanation.generate_high_confidence_explanation(_state("high"))

    assert result == {"explanation": "explanation"}
    assert "Explain the allocation primarily as the optimizer's selected strategy" in captured["prompt"]
