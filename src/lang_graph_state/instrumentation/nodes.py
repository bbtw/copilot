from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.state import RetirementPlanState
from lang_graph_state.instrumentation.timing import timed

NodeCallable = (
    Callable[[RetirementPlanState], dict[str, Any]]
    | Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]
)


def instrument_node(name: str, node: NodeCallable) -> NodeCallable:
    """Wrap a graph node so its start/finish logs include the keys it writes."""
    return timed(name, finish_attrs=lambda result: {"writes": sorted(result)})(node)
