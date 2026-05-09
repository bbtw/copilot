from collections.abc import Callable
from typing import Any

from services.explanation import ExplanationService
from state import RetirementPlanState


def build_synthesize_explanation_node(service: ExplanationService) -> Callable[[RetirementPlanState], dict[str, Any]]:
    def synthesize_explanation(state: RetirementPlanState) -> dict[str, Any]:
        explanation = service.synthesize(
            standard=state.standard_analysis,
            accumulation=state.accumulation_analysis,
            withdrawal=state.withdrawal_analysis,
            confidence_band=state.confidence_band,
        )
        return {"explanation": explanation}

    return synthesize_explanation
