from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.gateway import GatewayClient


def build_synthesize_explanation_node(
    client: GatewayClient,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def synthesize_explanation(state: GraphState) -> dict[str, Any]:
        explanation = await client.acomplete("", system="")
        return {"explanation": explanation}

    return synthesize_explanation
