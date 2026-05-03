from collections.abc import Callable
from typing import Any

from models import ConfidenceBand, PlanExplanationRequest
from services.explanation import ExplanationService
from state import RetirementPlanState


def build_explanation_node(
        expected_band: ConfidenceBand,
        service: ExplanationService,
) -> Callable[[RetirementPlanState], dict[str, Any]]:
    def generate_explanation(state: RetirementPlanState) -> dict[str, Any]:
        return _generate_explanation(state, expected_band=expected_band, service=service)

    return generate_explanation


def _generate_explanation(
        state: RetirementPlanState,
        *,
        expected_band: ConfidenceBand,
        service: ExplanationService,
) -> dict[str, Any]:
    actual_band = state.confidence_band
    # The graph route is the explanation policy. A mismatch means the graph
    # wiring or state was corrupted, so fail before generating client text.
    if actual_band != expected_band:
        raise ValueError(
            f"Explanation route mismatch: routed to {expected_band!r}, "
            f"but state confidence_band is {actual_band!r}"
        )

    # Keep LangGraph-specific state access in the node wrapper; the service
    # receives a stable domain request that production integrations can reuse.
    request = PlanExplanationRequest(
        customer_profile=state.customer_profile,
        contribution_allocation=state.contribution_allocation,
        projected_wealth=state.projected_wealth,
        wealth_distribution=state.wealth_distribution,
        confidence_score=state.confidence_score,
        confidence_band=expected_band,
        optimization_diagnostics=state.optimization_diagnostics,
    )

    return {"explanation": service.generate(request)}
