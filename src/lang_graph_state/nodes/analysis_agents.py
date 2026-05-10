import logging
from collections.abc import Awaitable, Callable
from typing import Any

from lang_graph_state.domain.models import AnalysisKind, PlanAnalysisSection, PlanExplanationRequest
from lang_graph_state.services.explanation import ExplanationService
from lang_graph_state.services.lp_sensitivity import run_default_lp_sensitivity_probes
from lang_graph_state.services.mc_sensitivity import run_default_mc_sensitivity_probes
from lang_graph_state.domain.state import RetirementPlanState

logger = logging.getLogger(__name__)


def _build_analysis_node(
    *,
    analysis_kind: AnalysisKind,
    service_call: Callable[[PlanExplanationRequest, list[str]], Awaitable[str]],
    build_probes: Callable[[RetirementPlanState], list[str]],
) -> Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    async def run_analysis(state: RetirementPlanState) -> dict[str, Any]:
        request = _request_from_state(state)
        probe_results = build_probes(state)
        logger.info(
            "Analysis branch start analysis_kind=%s probe_count=%s",
            analysis_kind,
            len(probe_results),
        )
        output = await service_call(request, probe_results)
        logger.info(
            "Analysis branch finish analysis_kind=%s output_chars=%s",
            analysis_kind,
            len(output),
        )
        return {"analysis_sections": [PlanAnalysisSection(kind=analysis_kind, content=output)]}
    return run_analysis


def _request_from_state(state: RetirementPlanState) -> PlanExplanationRequest:
    return PlanExplanationRequest(
        customer_profile=state.customer_profile,
        contribution_allocation=state.contribution_allocation,
        projected_wealth=state.projected_wealth,
        wealth_distribution=state.wealth_distribution,
        confidence_score=state.confidence_score,
        confidence_band=state.confidence_band,
        optimization_diagnostics=state.optimization_diagnostics,
    )


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
