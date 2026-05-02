from unittest.mock import MagicMock

from models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    OptimizationConstraintDiagnostic,
    OptimizationDiagnostics,
    OptimizationVariableDiagnostic,
    PlanExplanationRequest,
    WealthDistribution,
)
from services.explanation import ExplanationService
from services.llm import GatewayClient


def _mock_gateway(response: str = "service explanation") -> MagicMock:
    mock = MagicMock(spec=GatewayClient)
    mock.complete.return_value = response
    return mock


def _request(confidence_band: str = "low", diagnostics=None) -> PlanExplanationRequest:
    return PlanExplanationRequest(
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
            pre50={**{a: 0.0 for a in ACCOUNT_TYPES}, "hsa": 4_150},
            post50={**{a: 0.0 for a in ACCOUNT_TYPES}, "401k": 30_000},
        ),
        projected_wealth=1_000_000,
        wealth_distribution=WealthDistribution(p10=600_000, p50=900_000, p90=1_300_000),
        confidence_score=0.35,
        confidence_band=confidence_band,
        optimization_diagnostics=diagnostics,
    )


def test_generate_uses_injected_gateway_and_confidence_guidance():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    result = service.generate(_request("low"))

    prompt, kwargs = gateway.complete.call_args.args[0], gateway.complete.call_args.kwargs
    assert result == "service explanation"
    assert "certified financial planner" in kwargs["system"]
    assert "Do not present the plan as likely to meet the client's retirement goal" in prompt
    assert "Internal confidence band: low" in prompt
    assert "hsa: $4,150/year" in prompt


def test_generate_formats_optimizer_diagnostics_as_explainability_input():
    diagnostics = OptimizationDiagnostics(
        solver="CLARABEL",
        status="optimal",
        objective_value=1_000_000,
        variables=[
            OptimizationVariableDiagnostic(
                name="pre50_hsa",
                phase="pre50",
                account="hsa",
                value=4_150,
                objective_coefficient=8.50,
                reduced_cost=None,
                at_lower_bound=False,
            ),
            OptimizationVariableDiagnostic(
                name="pre50_roth_ira",
                phase="pre50",
                account="roth_ira",
                value=0,
                objective_coefficient=7.25,
                reduced_cost=0.42,
                at_lower_bound=True,
            ),
        ],
        constraints=[
            OptimizationConstraintDiagnostic(
                name="pre50_savings_capacity",
                category="budget",
                phase="pre50",
                slack=0,
                dual_value=1.25,
                binding=True,
            ),
        ],
    )
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    service.generate(_request("high", diagnostics=diagnostics))

    prompt = gateway.complete.call_args.args[0]
    assert "Solver: CLARABEL" in prompt
    assert "pre50_hsa: $4,150/year" in prompt
    assert "pre50_savings_capacity: slack $0.00, shadow price 1.25" in prompt
    assert "pre50_roth_ira: reduced-cost signal 0.42" in prompt
    assert "Explain the allocation primarily as the optimizer's selected strategy" in prompt
