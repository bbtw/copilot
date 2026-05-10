from typing import Any

from lang_graph_state.domain.state import RetirementPlanState


def format_output(state: RetirementPlanState) -> dict[str, Any]:
    return {"result": state.to_result()}
