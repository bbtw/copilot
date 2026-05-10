from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import AnalysisSection
from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.gateway import GatewayClient


def build_section_a_node(client: GatewayClient) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_section_a(state: GraphState) -> dict[str, Any]:
        content = await client.acomplete("", system="")
        return {"analysis_sections": [AnalysisSection(kind="section_a", content=content)]}

    return run_section_a


def build_section_b_node(client: GatewayClient) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_section_b(state: GraphState) -> dict[str, Any]:
        content = await client.acomplete("", system="")
        return {"analysis_sections": [AnalysisSection(kind="section_b", content=content)]}

    return run_section_b


def build_section_c_node(client: GatewayClient) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_section_c(state: GraphState) -> dict[str, Any]:
        content = await client.acomplete("", system="")
        return {"analysis_sections": [AnalysisSection(kind="section_c", content=content)]}

    return run_section_c
