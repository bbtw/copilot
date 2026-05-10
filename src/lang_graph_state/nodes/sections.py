from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import AnalysisKind, AnalysisPayload, AnalysisSection
from lang_graph_state.domain.state import GraphState
from lang_graph_state.services.explanation import ExplanationService


def _build_section_node(
    kind: AnalysisKind,
    service: ExplanationService,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    async def run_section(state: GraphState) -> dict[str, Any]:
        content = await service.aanalyze(kind, AnalysisPayload())
        return {"analysis_sections": [AnalysisSection(kind=kind, content=content)]}

    return run_section


def build_section_a_node(service: ExplanationService) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    return _build_section_node("section_a", service)


def build_section_b_node(service: ExplanationService) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    return _build_section_node("section_b", service)


def build_section_c_node(service: ExplanationService) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    return _build_section_node("section_c", service)
