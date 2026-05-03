from pydantic import BaseModel
from models import (
    CustomerProfile,
    ConfidenceBand,
    ContributionAllocation,
    OptimizationDiagnostics,
    WealthDistribution,
    RetirementPlanResult,
)


class RetirementPlanState(BaseModel):
    customer_profile: CustomerProfile | None = None
    contribution_allocation: ContributionAllocation | None = None
    optimization_diagnostics: OptimizationDiagnostics | None = None
    projected_wealth: float = 0.0
    wealth_distribution: WealthDistribution | None = None
    confidence_score: float = 0.0
    confidence_band: ConfidenceBand | None = None
    explanation: str = ""
    result: RetirementPlanResult | None = None
