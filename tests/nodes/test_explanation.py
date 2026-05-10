import asyncio

from lang_graph_state.domain.models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    PlanAnalysisSection,
    WealthDistribution,
)
from lang_graph_state.nodes.standard_analysis import build_standard_analysis_node
from lang_graph_state.nodes.synthesize_explanation import build_synthesize_explanation_node
from lang_graph_state.domain.state import RetirementPlanState


class FakeExplanationService:
    def __init__(self) -> None:
        self.standard_request = None
        self.synthesize_args = None

    async def astandard_analysis(self, request):
        self.standard_request = request
        return "standard analysis"

    async def asynthesize(self, standard, accumulation, withdrawal, confidence_band):
        self.synthesize_args = (standard, accumulation, withdrawal, confidence_band)
        return "synthesized explanation"


def _state(confidence_band: str = "high") -> RetirementPlanState:
    return RetirementPlanState(
        customer_profile=CustomerProfile(
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
        contribution_allocation=ContributionAllocation(
            pre50={a: 0.0 for a in ACCOUNT_TYPES},
            post50={a: 0.0 for a in ACCOUNT_TYPES},
        ),
        projected_wealth=1_000_000,
        wealth_distribution=WealthDistribution(p10=600_000, p50=900_000, p90=1_300_000),
        confidence_score=0.88,
        confidence_band=confidence_band,
        analysis_sections=[
            PlanAnalysisSection(kind="standard", content="plan overview"),
            PlanAnalysisSection(kind="accumulation", content="accumulation analysis"),
            PlanAnalysisSection(kind="withdrawal", content="withdrawal analysis"),
        ],
    )


def test_standard_analysis_node_adapts_state_to_service_request():
    service = FakeExplanationService()
    node = build_standard_analysis_node(service)

    result = asyncio.run(node(_state("high")))

    assert result == {
        "analysis_sections": [
            PlanAnalysisSection(kind="standard", content="standard analysis"),
        ],
    }
    assert service.standard_request.confidence_band == "high"
    assert service.standard_request.projected_wealth == 1_000_000
    assert service.standard_request.customer_profile.age == 42


def test_synthesize_explanation_node_passes_all_analyses_to_service():
    service = FakeExplanationService()
    node = build_synthesize_explanation_node(service)

    result = asyncio.run(node(_state("low")))

    assert result == {"explanation": "synthesized explanation"}
    standard, accumulation, withdrawal, band = service.synthesize_args
    assert standard == "plan overview"
    assert accumulation == "accumulation analysis"
    assert withdrawal == "withdrawal analysis"
    assert band == "low"
