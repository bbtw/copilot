from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.explanation import ExplanationService


def build_synthesize_explanation_node(
    service: ExplanationService,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def synthesize_explanation(state: GraphState) -> dict[str, Any]:
        explanation = await service.asynthesize(state.analysis_sections)
        return {"explanation": explanation}

    return synthesize_explanation
