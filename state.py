from typing import TypedDict
from models import (
    CustomerProfile,
    ConfidenceBand,
    ContributionAllocation,
    OptimizationDiagnostics,
    WealthDistribution,
    RetirementPlanResult,
)


class RetirementPlanState(TypedDict, total=False):
    customer_profile: CustomerProfile
    contribution_allocation: ContributionAllocation
    optimization_diagnostics: OptimizationDiagnostics
    projected_wealth: float
    wealth_distribution: WealthDistribution
    confidence_score: float
    confidence_band: ConfidenceBand
    explanation: str
    result: RetirementPlanResult
