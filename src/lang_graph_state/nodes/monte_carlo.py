from collections.abc import Callable
from typing import Any

import numpy as np

from lang_graph_state.services.mc_simulator import simulate
from lang_graph_state.domain.state import RetirementPlanState


def build_monte_carlo_node(rng: np.random.Generator | None = None) -> Callable[[RetirementPlanState], dict[str, Any]]:
    _rng = rng if rng is not None else np.random.default_rng()

    def run_monte_carlo(state: RetirementPlanState) -> dict[str, Any]:
        return simulate(state.customer_profile, state.contribution_allocation, rng=_rng)

    return run_monte_carlo
