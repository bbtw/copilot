from typing import Any

from models import RetirementPlanResult
from state import RetirementPlanState


def format_output(state: RetirementPlanState) -> dict[str, Any]:
    result = RetirementPlanResult(
        contribution_allocation=state.contribution_allocation,
        projected_wealth=state.projected_wealth,
        wealth_distribution=state.wealth_distribution,
        confidence_score=state.confidence_score,
        confidence_band=state.confidence_band,
        explanation=state.explanation,
        optimization_diagnostics=state.optimization_diagnostics,
    )
    return {"result": result}
