from typing import Any

from lang_graph_state.services.lp_solver import solve_lp
from lang_graph_state.domain.state import RetirementPlanState


def run_lp_optimizer(state: RetirementPlanState) -> dict[str, Any]:
    return solve_lp(state.customer_profile)
