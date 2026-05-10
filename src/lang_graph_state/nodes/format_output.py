from typing import Any

from lang_graph_state.domain.models import FinalOutput
from lang_graph_state.domain.state import GraphState


def format_output(state: GraphState) -> dict[str, Any]:
    return {"final_output": FinalOutput(explanation=state.explanation)}
