from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import PlanAnalysisSection
from lang_graph_state.domain.state import RetirementPlanState
from lang_graph_state.services.explanation import ExplanationService


def build_standard_analysis_node(service: ExplanationService) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    async def run_standard_analysis(state: RetirementPlanState) -> dict[str, Any]:
        content = await service.astandard_analysis(state.to_explanation_request())
        return {"analysis_sections": [PlanAnalysisSection(kind="standard", content=content)]}

    return run_standard_analysis
