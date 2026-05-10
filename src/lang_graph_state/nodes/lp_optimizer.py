from typing import Any

from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.lp_solver import solve_lp


async def run_lp_optimizer(state: GraphState) -> dict[str, Any]:
    return {"lp_result": await solve_lp(state.profile)}
