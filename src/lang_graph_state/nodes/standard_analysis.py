from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import PlanAnalysisSection, PlanExplanationRequest
from lang_graph_state.services.explanation import ExplanationService
from lang_graph_state.domain.state import RetirementPlanState


def build_standard_analysis_node(service: ExplanationService) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    async def run_standard_analysis(state: RetirementPlanState) -> dict[str, Any]:
        request = PlanExplanationRequest(
            customer_profile=state.customer_profile,
            contribution_allocation=state.contribution_allocation,
            projected_wealth=state.projected_wealth,
            wealth_distribution=state.wealth_distribution,
            confidence_score=state.confidence_score,
            confidence_band=state.confidence_band,
            optimization_diagnostics=state.optimization_diagnostics,
        )
        return {
            "analysis_sections": [
                PlanAnalysisSection(kind="standard", content=await service.astandard_analysis(request)),
            ],
        }

    return run_standard_analysis
