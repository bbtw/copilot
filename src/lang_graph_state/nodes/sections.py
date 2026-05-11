"""
Fan-out analysis nodes. Each runs concurrently after run_monte_carlo and writes one AnalysisSection
to the shared reducer-backed state field. To add a new analysis type, add a new builder here.
"""
from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import AnalysisSection
from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.gateway import GatewayClient


def build_standard_explanation_node(client: GatewayClient) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_standard_explanation(state: GraphState) -> dict[str, Any]:
        content = await client.acomplete("", system="")
        return {"analysis_sections": [AnalysisSection(kind="standard_explanation", content=content)]}

    return run_standard_explanation


def build_contribution_explanation_node(client: GatewayClient) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_contribution_explanation(state: GraphState) -> dict[str, Any]:
        content = await client.acomplete("", system="")
        return {"analysis_sections": [AnalysisSection(kind="contribution_explanation", content=content)]}

    return run_contribution_explanation


def build_withdrawal_explanation_node(client: GatewayClient) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_withdrawal_explanation(state: GraphState) -> dict[str, Any]:
        content = await client.acomplete("", system="")
        return {"analysis_sections": [AnalysisSection(kind="withdrawal_explanation", content=content)]}

    return run_withdrawal_explanation
