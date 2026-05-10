from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import AnalysisKind, PlanAnalysisSection, PlanExplanationRequest
from lang_graph_state.domain.state import RetirementPlanState
from lang_graph_state.services.explanation import ExplanationService
from lang_graph_state.services.lp_sensitivity import run_default_lp_sensitivity_probes
from lang_graph_state.services.mc_sensitivity import run_default_mc_sensitivity_probes


def _build_analysis_node(
    *,
    analysis_kind: AnalysisKind,
    service_call: Callable[[PlanExplanationRequest, list[str]], Awaitable[str]],
    build_probes: Callable[[RetirementPlanState], list[str]],
) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    async def run_analysis(state: RetirementPlanState) -> dict[str, Any]:
        content = await service_call(state.to_explanation_request(), build_probes(state))
        return {"analysis_sections": [PlanAnalysisSection(kind=analysis_kind, content=content)]}
    return run_analysis


def build_accumulation_node(service: ExplanationService) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    return _build_analysis_node(
        analysis_kind="accumulation",
        service_call=service.aaccumulation_analysis,
        build_probes=lambda state: run_default_lp_sensitivity_probes(state.customer_profile),
    )


def build_withdrawal_node(service: ExplanationService) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    return _build_analysis_node(
        analysis_kind="withdrawal",
        service_call=service.awithdrawal_analysis,
        build_probes=lambda state: run_default_mc_sensitivity_probes(state.customer_profile, state.contribution_allocation),
    )
