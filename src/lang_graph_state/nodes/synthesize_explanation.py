"""
Fan-in node. Receives all analysis sections once the parallel branches have completed and been merged,
then synthesizes them into a single explanation.
"""
from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.gateway import GatewayClient


def build_synthesize_explanation_node(
    client: GatewayClient,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def synthesize_explanation(state: GraphState) -> dict[str, Any]:
        # LangGraph only schedules this node after all three fan-out branches have written and been merged by the reducer.
        explanation = await client.acomplete("", system="")
        return {"explanation": explanation}

    return synthesize_explanation
