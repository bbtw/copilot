from collections.abc import Callable
from typing import Any

from models import PlanExplanationRequest
from services.explanation import ExplanationService
from state import RetirementPlanState


def build_standard_analysis_node(service: ExplanationService) -> Callable[[RetirementPlanState], dict[str, Any]]:
    def run_standard_analysis(state: RetirementPlanState) -> dict[str, Any]:
        request = PlanExplanationRequest(
            customer_profile=state.customer_profile,
            contribution_allocation=state.contribution_allocation,
            projected_wealth=state.projected_wealth,
            wealth_distribution=state.wealth_distribution,
            confidence_score=state.confidence_score,
            confidence_band=state.confidence_band,
            optimization_diagnostics=state.optimization_diagnostics,
        )
        return {"standard_analysis": service.standard_analysis(request)}

    return run_standard_analysis
