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
    """
    The overall state schema for the retirement plan graph.
    
    Note on Updates: While nodes receive the full `RetirementPlanState` object, 
    they should return a standard `dict` containing only the fields they wish 
    to update. LangGraph idiomatically accepts partial dictionaries and handles 
    applying those updates back into the Pydantic state object behind the scenes.
    """
    customer_profile: CustomerProfile | None = None
    contribution_allocation: ContributionAllocation | None = None
    optimization_diagnostics: OptimizationDiagnostics | None = None
    projected_wealth: float = 0.0
    wealth_distribution: WealthDistribution | None = None
    confidence_score: float = 0.0
    confidence_band: ConfidenceBand | None = None
    explanation: str = ""
    result: RetirementPlanResult | None = None
