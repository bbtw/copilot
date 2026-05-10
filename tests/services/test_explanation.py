import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from lang_graph_state.domain.models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    OptimizationConstraintDiagnostic,
    OptimizationDiagnostics,
    OptimizationVariableDiagnostic,
    PlanExplanationRequest,
    WealthDistribution,
)
from lang_graph_state.services.explanation import ExplanationService
from lang_graph_state.services.llm import GatewayClient


def _mock_gateway(response: str = "service explanation") -> MagicMock:
    mock = MagicMock(spec=GatewayClient)
    mock.complete.return_value = response
    mock.acomplete = AsyncMock(return_value=response)
    return mock


def test_gateway_rejects_direct_local_ollama_url(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("LLM_MODEL_ID", "llama3.2")

    with pytest.raises(ValueError, match="FastAPI LLM gateway"):
        GatewayClient()


def test_gateway_exposes_async_completion_method():
    assert callable(GatewayClient.acomplete)


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


def test_standard_analysis_uses_injected_gateway():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    result = service.standard_analysis(_request("low"))

    prompt, kwargs = gateway.complete.call_args.args[0], gateway.complete.call_args.kwargs
    assert result == "service explanation"
    assert "certified financial planner" in kwargs["system"]
    assert "hsa: $4,150/year" in prompt


def test_standard_analysis_includes_optimizer_diagnostics_in_prompt():
    diagnostics = OptimizationDiagnostics(
        solver="CLARABEL",
        status="optimal",
        objective_value=1_000_000,
        variables=[
            OptimizationVariableDiagnostic(
                name="pre50_hsa", phase="pre50", account="hsa", value=4_150,
                objective_coefficient=8.50, reduced_cost=None, at_lower_bound=False,
            ),
            OptimizationVariableDiagnostic(
                name="pre50_roth_ira", phase="pre50", account="roth_ira", value=0,
                objective_coefficient=7.25, reduced_cost=0.42, at_lower_bound=True,
            ),
        ],
        constraints=[
            OptimizationConstraintDiagnostic(
                name="pre50_savings_capacity", category="budget", phase="pre50",
                slack=0, dual_value=1.25, binding=True,
            ),
        ],
    )
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    service.standard_analysis(_request("high", diagnostics=diagnostics))

    prompt = gateway.complete.call_args.args[0]
    assert "pre50_hsa: $4,150/year" in prompt
    assert "pre50_savings_capacity: slack $0.00, shadow price 1.25" in prompt
    assert "pre50_roth_ira: reduced-cost signal 0.42" in prompt


def test_synthesize_includes_confidence_guidance_in_prompt():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    service.synthesize("overview", "accumulation", "withdrawal", "low")

    prompt = gateway.complete.call_args.args[0]
    assert "overview" in prompt
    assert "accumulation" in prompt
    assert "withdrawal" in prompt
    assert "Do not present the plan as likely to meet" in prompt


def test_synthesize_high_confidence_guidance():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    service.synthesize("overview", "accumulation", "withdrawal", "high")

    prompt = gateway.complete.call_args.args[0]
    assert "Explain the allocation primarily as the optimizer's selected strategy" in prompt


def test_async_standard_analysis_uses_async_gateway():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    result = asyncio.run(service.astandard_analysis(_request("low")))

    prompt, kwargs = gateway.acomplete.call_args.args[0], gateway.acomplete.call_args.kwargs
    assert result == "service explanation"
    assert "certified financial planner" in kwargs["system"]
    assert "hsa: $4,150/year" in prompt


def test_async_accumulation_analysis_uses_async_gateway():
    gateway = _mock_gateway("accumulation")
    service = ExplanationService(llm=gateway)

    result = asyncio.run(service.aaccumulation_analysis(_request("high"), ["Scenario: lower expenses"]))

    prompt, kwargs = gateway.acomplete.call_args.args[0], gateway.acomplete.call_args.kwargs
    assert result == "accumulation"
    assert kwargs["max_tokens"] == 1536
    assert "accumulation phase" in kwargs["system"]
    assert "Scenario: lower expenses" in prompt


def test_async_withdrawal_analysis_uses_async_gateway():
    gateway = _mock_gateway("withdrawal")
    service = ExplanationService(llm=gateway)

    result = asyncio.run(service.awithdrawal_analysis(_request("medium"), ["Scenario: longer retirement"]))

    prompt, kwargs = gateway.acomplete.call_args.args[0], gateway.acomplete.call_args.kwargs
    assert result == "withdrawal"
    assert kwargs["max_tokens"] == 1536
    assert "decumulation" in kwargs["system"]
    assert "Scenario: longer retirement" in prompt


def test_async_synthesize_uses_async_gateway():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)

    asyncio.run(service.asynthesize("overview", "accumulation", "withdrawal", "low"))

    prompt, kwargs = gateway.acomplete.call_args.args[0], gateway.acomplete.call_args.kwargs
    assert kwargs["max_tokens"] == 1536
    assert "Do not present the plan as likely to meet" in prompt
