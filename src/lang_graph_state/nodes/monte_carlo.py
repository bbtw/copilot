from typing import Any

from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.mc_simulator import simulate


async def run_monte_carlo(state: GraphState) -> dict[str, Any]:
    return {"mc_result": await simulate(state.profile, state.lp_result)}
