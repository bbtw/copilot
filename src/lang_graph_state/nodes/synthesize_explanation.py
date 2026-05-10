from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.services.explanation import ExplanationService
from lang_graph_state.domain.state import RetirementPlanState


def build_synthesize_explanation_node(service: ExplanationService) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    async def synthesize_explanation(state: RetirementPlanState) -> dict[str, Any]:
        # The graph wiring guarantees all three parallel branches have joined
        # before this node runs, so these keys should be present.
        analyses = state.analysis_by_kind()
        explanation = await service.asynthesize(
            standard=analyses["standard"],
            accumulation=analyses["accumulation"],
            withdrawal=analyses["withdrawal"],
            confidence_band=state.confidence_band,
        )
        return {"explanation": explanation}

    return synthesize_explanation
