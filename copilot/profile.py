"""The Profile: everything the solver needs about the User (see CONTEXT.md)."""

from dataclasses import dataclass
from enum import Enum

from .plan import Objective
from .taxdata import (
    DEFAULT_CAPITAL_GAINS_RATE,
    DEFAULT_CONTRIBUTION_LIMIT,
    DEFAULT_REAL_RETURN,
    RMD_START_AGE,
    SS_TAXABLE_FRACTION,
    FilingStatus,
)


class IncomeKind(Enum):
    SOCIAL_SECURITY = "social_security"
    PENSION = "pension"


@dataclass(frozen=True)
class IncomeStream:
    kind: IncomeKind
    annual_amount: float
    start_age: int

    @property
    def taxable_fraction(self) -> float:
        return SS_TAXABLE_FRACTION if self.kind is IncomeKind.SOCIAL_SECURITY else 1.0


@dataclass(frozen=True)
class Profile:
    current_age: int
    retirement_age: int
    horizon_age: int
    filing_status: FilingStatus
    gross_income: float
    savings_capacity: float
    spending_need: float
    balance_traditional: float
    balance_roth: float
    balance_taxable: float
    income_streams: tuple[IncomeStream, ...] = ()
    employer_match: float = 0.0
    expected_return: float = DEFAULT_REAL_RETURN
    capital_gains_rate: float = DEFAULT_CAPITAL_GAINS_RATE
    contribution_limit: float = DEFAULT_CONTRIBUTION_LIMIT


_OPTIONAL_FIELDS = (
    "employer_match", "expected_return", "capital_gains_rate", "contribution_limit",
)


def from_solve_args(args: dict) -> tuple[Profile, Objective]:
    """Build the run inputs — one Profile + one Objective — from raw solve_plan
    tool arguments. Raises KeyError/ValueError on missing or malformed fields."""
    objective = Objective(args["objective"])
    kwargs = {
        "current_age": int(args["current_age"]),
        "retirement_age": int(args["retirement_age"]),
        "horizon_age": int(args["horizon_age"]),
        "filing_status": FilingStatus(args["filing_status"]),
        "gross_income": float(args["gross_income"]),
        "savings_capacity": float(args["savings_capacity"]),
        "spending_need": float(args["spending_need"]),
        "balance_traditional": float(args["balance_traditional"]),
        "balance_roth": float(args["balance_roth"]),
        "balance_taxable": float(args["balance_taxable"]),
        "income_streams": tuple(
            IncomeStream(
                kind=IncomeKind(s["kind"]),
                annual_amount=float(s["annual_amount"]),
                start_age=int(s["start_age"]),
            )
            for s in args.get("income_streams", [])
        ),
    }
    for field in _OPTIONAL_FIELDS:
        if args.get(field) is not None:
            kwargs[field] = float(args[field])
    return Profile(**kwargs), objective


def validate(profile: Profile) -> None:
    """Raise ValueError with a chat-presentable message if the Profile is unusable."""
    p = profile
    if not 18 <= p.current_age <= p.retirement_age:
        raise ValueError("current_age must be at least 18 and no greater than retirement_age")
    if p.retirement_age > RMD_START_AGE:
        raise ValueError(f"v1 supports retirement no later than the RMD age ({RMD_START_AGE})")
    if p.horizon_age <= p.retirement_age:
        raise ValueError("horizon_age must be greater than retirement_age")
    if p.horizon_age > 120:
        raise ValueError("horizon_age must be 120 or less")
    for name in (
        "gross_income", "savings_capacity", "spending_need",
        "balance_traditional", "balance_roth", "balance_taxable",
        "employer_match", "contribution_limit",
    ):
        if getattr(p, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if p.expected_return <= -1.0:
        raise ValueError("expected_return must be greater than -100%")
    if not 0.0 <= p.capital_gains_rate < 1.0:
        raise ValueError("capital_gains_rate must be in [0, 1)")
    for stream in p.income_streams:
        if stream.annual_amount < 0:
            raise ValueError("income stream amounts cannot be negative")
        if not p.current_age <= stream.start_age <= p.horizon_age:
            raise ValueError("income stream start_age must fall within the plan years")
