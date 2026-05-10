from typing import Literal, get_args

from pydantic import BaseModel, Field

AccountType = Literal["401k", "roth_401k", "traditional_ira", "roth_ira", "hsa", "taxable_brokerage"]
ACCOUNT_TYPES: list[AccountType] = list(get_args(AccountType))

FilingStatus = Literal["single", "married_filing_jointly", "married_filing_separately", "head_of_household"]
ConfidenceBand = Literal["low", "medium", "high"]
AnalysisKind = Literal["standard", "accumulation", "withdrawal"]
ANALYSIS_SECTION_ORDER: dict[AnalysisKind, int] = {
    "standard": 0,
    "accumulation": 1,
    "withdrawal": 2,
}


def classify_confidence(score: float) -> ConfidenceBand:
    if score < 0.70:
        return "low"
    if score < 0.85:
        return "medium"
    return "high"


class CustomerProfile(BaseModel):
    age: int
    retirement_age: int
    annual_income: float
    annual_expenses: float
    retirement_annual_expenses: float
    retirement_years_to_plan: int
    expected_retirement_income: float
    balances: dict[str, float]
    employer_match_rate: float
    employer_match_cap: float
    hdhp_enrolled: bool
    current_marginal_tax_rate: float
    assumed_retirement_marginal_tax_rate: float
    filing_status: FilingStatus

    @property
    def savings_capacity(self) -> float:
        return self.annual_income - self.annual_expenses

    @property
    def years_to_retirement(self) -> int:
        return self.retirement_age - self.age

    @property
    def pre50_years(self) -> int:
        return max(0, 50 - self.age)

    @property
    def post50_years(self) -> int:
        return max(0, self.retirement_age - max(self.age, 50))


class ContributionAllocation(BaseModel):
    pre50: dict[str, float] = Field(default_factory=dict)
    post50: dict[str, float] = Field(default_factory=dict)

    def for_age(self, age: int) -> dict[str, float]:
        return self.post50 if age >= 50 else self.pre50


class WealthDistribution(BaseModel):
    p10: float
    p50: float
    p90: float


class OptimizationVariableDiagnostic(BaseModel):
    name: str
    phase: str
    account: str
    value: float
    objective_coefficient: float
    reduced_cost: float | None
    at_lower_bound: bool


class OptimizationConstraintDiagnostic(BaseModel):
    name: str
    category: str
    phase: str | None
    slack: float
    dual_value: float | None
    binding: bool


class OptimizationDiagnostics(BaseModel):
    solver: str
    status: str
    objective_value: float
    variables: list[OptimizationVariableDiagnostic] = Field(default_factory=list)
    constraints: list[OptimizationConstraintDiagnostic] = Field(default_factory=list)


class PlanAnalysisSection(BaseModel):
    """One analysis branch output collected by the parent graph reducer."""

    kind: AnalysisKind
    content: str


def merge_analysis_sections(
    current: list[PlanAnalysisSection] | None,
    updates: list[PlanAnalysisSection] | None,
) -> list[PlanAnalysisSection]:
    """
    LangGraph reducer for the parallel LLM fan-out.

    The three analysis nodes all write to the same state key. A reducer makes
    that concurrent write explicit and keeps retries/idempotent replays from
    duplicating sections with the same kind.
    """
    by_kind = {section.kind: section for section in current or []}
    for section in updates or []:
        by_kind[section.kind] = section
    return sorted(by_kind.values(), key=lambda section: ANALYSIS_SECTION_ORDER[section.kind])


class PlanExplanationRequest(BaseModel):
    """Service contract for the explanation service, decoupled from LangGraph state shape."""
    customer_profile: CustomerProfile
    contribution_allocation: ContributionAllocation
    projected_wealth: float
    wealth_distribution: WealthDistribution
    confidence_score: float
    confidence_band: ConfidenceBand
    optimization_diagnostics: OptimizationDiagnostics | None = None


class RetirementPlanResult(BaseModel):
    contribution_allocation: ContributionAllocation
    projected_wealth: float
    wealth_distribution: WealthDistribution
    confidence_score: float
    confidence_band: ConfidenceBand
    explanation: str
    optimization_diagnostics: OptimizationDiagnostics | None = None

    def print_summary(self) -> None:
        print("=" * 60)
        print("RETIREMENT PLAN SUMMARY")
        print("=" * 60)
        print("\nAnnual Contributions (Pre-50):")
        for acct, amt in self.contribution_allocation.pre50.items():
            if amt > 0:
                print(f"  {acct:25s} ${amt:>10,.0f}")
        print("\nAnnual Contributions (Post-50 / Catch-up):")
        for acct, amt in self.contribution_allocation.post50.items():
            if amt > 0:
                print(f"  {acct:25s} ${amt:>10,.0f}")
        print(f"\nProjected After-Tax Wealth at Retirement: ${self.projected_wealth:>12,.0f}")
        print(f"\nMonte Carlo percentiles (today's dollars):")
        print(f"  10th percentile: ${self.wealth_distribution.p10:>12,.0f}")
        print(f"  50th percentile: ${self.wealth_distribution.p50:>12,.0f}")
        print(f"  90th percentile: ${self.wealth_distribution.p90:>12,.0f}")
        print(f"  Confidence score: {self.confidence_score:.1%}")
        print(f"  Confidence band: {self.confidence_band}")
        if self.optimization_diagnostics:
            binding = [c for c in self.optimization_diagnostics.constraints if c.binding]
            print("\nBinding Optimizer Constraints:")
            for c in binding[:8]:
                dual = "n/a" if c.dual_value is None else f"{c.dual_value:,.2f}"
                phase = "" if c.phase is None else f" ({c.phase})"
                print(f"  {c.name}{phase:12s} dual={dual}")
        print(f"\nExplanation:\n{self.explanation}")
        print("=" * 60)
