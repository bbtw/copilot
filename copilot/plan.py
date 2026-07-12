"""The Plan: the solver's output and the only source of financial numbers (ADR-0002)."""

from dataclasses import dataclass
from enum import Enum


class Objective(Enum):
    WEALTH_AT_RETIREMENT = "wealth_at_retirement"
    TERMINAL_WEALTH = "terminal_wealth"


@dataclass(frozen=True)
class YearRow:
    age: int
    phase: str  # "accumulation" | "decumulation"
    contrib_traditional: float
    contrib_roth: float
    contrib_taxable: float
    employer_match: float
    conversion: float
    withdraw_traditional: float
    withdraw_roth: float
    withdraw_taxable: float
    taxable_income: float
    tax: float
    balance_traditional: float
    balance_roth: float
    balance_taxable: float


@dataclass(frozen=True)
class Plan:
    objective: Objective
    objective_value: float  # After-Tax Value at the measurement year
    rows: tuple[YearRow, ...]


@dataclass(frozen=True)
class Infeasible:
    reason: str


def to_summary(plan: Plan) -> dict:
    """Compact dict handed to the LLM as the tool result (ADR-0002: it narrates only this)."""
    accumulation = [r for r in plan.rows if r.phase == "accumulation"]
    decumulation = [r for r in plan.rows if r.phase == "decumulation"]
    conversion_ages = [r.age for r in plan.rows if r.conversion > 1.0]
    summary: dict = {
        "objective": plan.objective.value,
        "objective_value_after_tax": round(plan.objective_value),
        "units": "today's dollars (real); after-tax value",
        "total_lifetime_tax": round(sum(r.tax for r in plan.rows)),
        "conversions": {
            "total": round(sum(r.conversion for r in plan.rows)),
            "ages": conversion_ages,
        },
        "decumulation_totals": {
            "withdraw_traditional": round(sum(r.withdraw_traditional for r in decumulation)),
            "withdraw_roth": round(sum(r.withdraw_roth for r in decumulation)),
            "withdraw_taxable": round(sum(r.withdraw_taxable for r in decumulation)),
        },
        "balances_at_horizon": {
            "traditional": round(plan.rows[-1].balance_traditional),
            "roth": round(plan.rows[-1].balance_roth),
            "taxable": round(plan.rows[-1].balance_taxable),
        },
    }
    if accumulation:
        summary["accumulation_totals"] = {
            "years": len(accumulation),
            "contrib_traditional": round(sum(r.contrib_traditional for r in accumulation)),
            "contrib_roth": round(sum(r.contrib_roth for r in accumulation)),
            "contrib_taxable": round(sum(r.contrib_taxable for r in accumulation)),
            "employer_match": round(sum(r.employer_match for r in accumulation)),
        }
        last_working = accumulation[-1]
        summary["balances_at_retirement"] = {
            "traditional": round(last_working.balance_traditional),
            "roth": round(last_working.balance_roth),
            "taxable": round(last_working.balance_taxable),
        }
    return summary
