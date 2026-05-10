from typing import Any

from lang_graph_state.domain.models import Profile
from lang_graph_state.domain.state import GraphState


def load_profile(state: GraphState) -> dict[str, Any]:
    return {"profile": Profile()}
