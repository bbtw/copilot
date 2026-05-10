from typing import Annotated

from pydantic import BaseModel, Field
from lang_graph_state.domain.models import (
    AnalysisKind,
    CustomerProfile,
    ConfidenceBand,
    ContributionAllocation,
    OptimizationDiagnostics,
    PlanAnalysisSection,
    PlanExplanationRequest,
    WealthDistribution,
    RetirementPlanResult,
    merge_analysis_sections,
)


class RetirementPlanState(BaseModel):
    """
    Accumulated LangGraph state passed through the retirement planning graph.

    Nodes receive the full state object and return a dict of only the fields they
    own. Most fields use LangGraph's default "last write wins" reducer because a
    single node owns them. The parallel analysis branches share `analysis_sections`,
    so that field has an explicit reducer.
    """
    customer_profile: CustomerProfile | None = None
    contribution_allocation: ContributionAllocation | None = None
    optimization_diagnostics: OptimizationDiagnostics | None = None
    projected_wealth: float = 0.0
    wealth_distribution: WealthDistribution | None = None
    confidence_score: float = 0.0
    confidence_band: ConfidenceBand | None = None
    analysis_sections: Annotated[list[PlanAnalysisSection], merge_analysis_sections] = Field(
        default_factory=list,
    )
    explanation: str = ""
    result: RetirementPlanResult | None = None

    def analysis_by_kind(self) -> dict[AnalysisKind, str]:
        return {section.kind: section.content for section in self.analysis_sections}

    def to_explanation_request(self) -> PlanExplanationRequest:
        return PlanExplanationRequest(
            customer_profile=self.customer_profile,
            contribution_allocation=self.contribution_allocation,
            projected_wealth=self.projected_wealth,
            wealth_distribution=self.wealth_distribution,
            confidence_score=self.confidence_score,
            confidence_band=self.confidence_band,
            optimization_diagnostics=self.optimization_diagnostics,
        )

    def to_result(self) -> RetirementPlanResult:
        return RetirementPlanResult(
            contribution_allocation=self.contribution_allocation,
            projected_wealth=self.projected_wealth,
            wealth_distribution=self.wealth_distribution,
            confidence_score=self.confidence_score,
            confidence_band=self.confidence_band,
            explanation=self.explanation,
            optimization_diagnostics=self.optimization_diagnostics,
        )
