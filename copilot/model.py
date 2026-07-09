"""The lifetime LP: one Profile + one Objective in, one Plan out (ADR-0001/0002).

Pure LP, no binaries: progressive brackets and the liquidation valuation are
convex piecewise-linear, so segment variables suffice. Everything is annual
and in real (today's) dollars.
"""

import gurobipy as gp
from gurobipy import GRB
from langsmith import traceable

from .plan import Infeasible, Objective, Plan, YearRow
from .profile import Profile, validate
from .taxdata import (
    EARLY_WITHDRAWAL_AGE,
    RMD_START_AGE,
    ordinary_segments,
    rmd_divisor,
    tax_on_ordinary_income,
)

# Free pip-install Gurobi license cap; checked before optimize (see grilling session)
GUROBI_TRIAL_LIMIT = 2000

_INFEASIBLE_REASON = (
    "The plan is infeasible under these inputs. Most likely the annual spending "
    "need cannot be met from the balances and guaranteed income provided — or, "
    "for early retirees, the taxable account cannot bridge spending until "
    f"retirement accounts unlock at age {EARLY_WITHDRAWAL_AGE}."
)


class ModelTooLargeError(RuntimeError):
    pass


def _segment_vars(m: gp.Model, segments: list[tuple[float, float]], tag: str) -> list[gp.Var]:
    return [
        m.addVar(ub=GRB.INFINITY if width == float("inf") else width, name=f"{tag}[{k}]")
        for k, (width, _) in enumerate(segments)
    ]


def _after_tax_value(
    m: gp.Model,
    segments: list[tuple[float, float]],
    traditional,
    roth,
    taxable,
    tag: str,
):
    """After-Tax Value of a balance triple: Roth and Taxable at face, Traditional
    net of a one-shot bracket-taxed liquidation (concave, so LP-safe to maximize)."""
    liquidation = _segment_vars(m, segments, f"liq_{tag}")
    m.addConstr(gp.quicksum(liquidation) == traditional, name=f"liq_total_{tag}")
    liquidation_tax = gp.quicksum(rate * v for v, (_, rate) in zip(liquidation, segments))
    return roth + taxable + traditional - liquidation_tax


@traceable(name="solve_plan")
def solve(profile: Profile, objective: Objective) -> Plan | Infeasible:
    """Build and solve the lifetime LP for one Profile and one Objective."""
    validate(profile)
    p = profile
    years = p.horizon_age - p.current_age
    ages = [p.current_age + t for t in range(years)]
    is_working = [age < p.retirement_age for age in ages]
    segments = ordinary_segments(p.filing_status)
    r = p.expected_return
    r_taxable = r * (1 - p.capital_gains_rate)  # Tax Drag
    gi_cash = [
        sum(s.annual_amount for s in p.income_streams if s.start_age <= age) for age in ages
    ]
    gi_taxable = [
        sum(
            s.annual_amount * s.taxable_fraction
            for s in p.income_streams
            if s.start_age <= age
        )
        for age in ages
    ]
    # Savings Capacity means "cash available if all saving were post-tax"; the
    # Traditional deduction benefit is endogenous relative to this baseline.
    baseline_tax = tax_on_ordinary_income(p.gross_income, p.filing_status)

    m = gp.Model("retirement")
    m.Params.OutputFlag = 0

    c_trad, c_roth, c_taxb = {}, {}, {}
    w_trad, w_roth, w_taxb, reinvest = {}, {}, {}, {}
    conv, b_trad, b_roth, b_taxb = {}, {}, {}, {}
    income_expr, tax_expr = {}, {}

    for t, age in enumerate(ages):
        conv[t] = m.addVar(name=f"conv[{t}]")
        if is_working[t]:
            c_trad[t] = m.addVar(name=f"c_trad[{t}]")
            c_roth[t] = m.addVar(name=f"c_roth[{t}]")
            c_taxb[t] = m.addVar(name=f"c_taxb[{t}]")
        else:
            unlocked = age >= EARLY_WITHDRAWAL_AGE
            w_trad[t] = m.addVar(ub=GRB.INFINITY if unlocked else 0.0, name=f"w_trad[{t}]")
            w_roth[t] = m.addVar(ub=GRB.INFINITY if unlocked else 0.0, name=f"w_roth[{t}]")
            w_taxb[t] = m.addVar(name=f"w_taxb[{t}]")
            reinvest[t] = m.addVar(name=f"reinvest[{t}]")
        b_trad[t] = m.addVar(name=f"b_trad[{t}]")
        b_roth[t] = m.addVar(name=f"b_roth[{t}]")
        b_taxb[t] = m.addVar(name=f"b_taxb[{t}]")

        segment_vars = _segment_vars(m, segments, f"inc[{t}]")
        income_expr[t] = (
            (p.gross_income if is_working[t] else 0.0)
            - c_trad.get(t, 0.0)
            + conv[t]
            + w_trad.get(t, 0.0)
            + gi_taxable[t]
        )
        m.addConstr(gp.quicksum(segment_vars) == income_expr[t], name=f"income[{t}]")
        tax_expr[t] = gp.quicksum(rate * v for v, (_, rate) in zip(segment_vars, segments))

        if is_working[t]:
            m.addConstr(
                c_trad[t] + c_roth[t] + c_taxb[t] + tax_expr[t]
                == p.savings_capacity + baseline_tax,
                name=f"budget[{t}]",
            )
            m.addConstr(
                c_trad[t] + c_roth[t] <= p.contribution_limit, name=f"limit[{t}]"
            )
        else:
            m.addConstr(
                w_trad[t] + w_roth[t] + w_taxb[t] + gi_cash[t] - tax_expr[t]
                == p.spending_need + reinvest[t],
                name=f"spending[{t}]",
            )

        prev_trad = b_trad[t - 1] if t > 0 else p.balance_traditional
        prev_roth = b_roth[t - 1] if t > 0 else p.balance_roth
        prev_taxb = b_taxb[t - 1] if t > 0 else p.balance_taxable
        if age >= RMD_START_AGE:
            m.addConstr(
                w_trad[t] >= (1.0 / rmd_divisor(age)) * prev_trad, name=f"rmd[{t}]"
            )
        match = p.employer_match if is_working[t] else 0.0
        m.addConstr(
            b_trad[t]
            == (prev_trad + c_trad.get(t, 0.0) + match - w_trad.get(t, 0.0) - conv[t])
            * (1 + r),
            name=f"bal_trad[{t}]",
        )
        m.addConstr(
            b_roth[t]
            == (prev_roth + c_roth.get(t, 0.0) + conv[t] - w_roth.get(t, 0.0)) * (1 + r),
            name=f"bal_roth[{t}]",
        )
        m.addConstr(
            b_taxb[t]
            == (prev_taxb + c_taxb.get(t, 0.0) + reinvest.get(t, 0.0) - w_taxb.get(t, 0.0))
            * (1 + r_taxable),
            name=f"bal_taxb[{t}]",
        )

    atv_terminal = _after_tax_value(
        m, segments, b_trad[years - 1], b_roth[years - 1], b_taxb[years - 1], "term"
    )
    if objective is Objective.WEALTH_AT_RETIREMENT:
        ret_idx = p.retirement_age - p.current_age - 1
        if ret_idx >= 0:
            measured = (b_trad[ret_idx], b_roth[ret_idx], b_taxb[ret_idx])
        else:  # already retired: measurement is the initial balances
            measured = (p.balance_traditional, p.balance_roth, p.balance_taxable)
        atv_retirement = _after_tax_value(m, segments, *measured, "ret")
        # Tiny terminal-wealth tiebreak so decumulation years get sensible values
        # instead of arbitrary feasible ones
        m.setObjective(atv_retirement + 1e-6 * atv_terminal, GRB.MAXIMIZE)
    else:
        m.setObjective(atv_terminal, GRB.MAXIMIZE)

    m.update()
    if m.NumVars > GUROBI_TRIAL_LIMIT or m.NumConstrs > GUROBI_TRIAL_LIMIT:
        raise ModelTooLargeError(
            f"Model has {m.NumVars} variables / {m.NumConstrs} constraints, which "
            f"exceeds the free Gurobi license limit of {GUROBI_TRIAL_LIMIT}. "
            "Shorten the plan horizon or install a full Gurobi license."
        )
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return Infeasible(reason=_INFEASIBLE_REASON)

    def value(x) -> float:
        if isinstance(x, gp.Var):
            return x.X
        if isinstance(x, (int, float)):
            return float(x)
        return x.getValue()

    rows = tuple(
        YearRow(
            age=age,
            phase="accumulation" if is_working[t] else "decumulation",
            contrib_traditional=value(c_trad.get(t, 0.0)),
            contrib_roth=value(c_roth.get(t, 0.0)),
            contrib_taxable=value(c_taxb.get(t, 0.0)),
            employer_match=p.employer_match if is_working[t] else 0.0,
            conversion=value(conv[t]),
            withdraw_traditional=value(w_trad.get(t, 0.0)),
            withdraw_roth=value(w_roth.get(t, 0.0)),
            withdraw_taxable=value(w_taxb.get(t, 0.0)),
            taxable_income=value(income_expr[t]),
            tax=value(tax_expr[t]),
            balance_traditional=value(b_trad[t]),
            balance_roth=value(b_roth[t]),
            balance_taxable=value(b_taxb[t]),
        )
        for t, age in enumerate(ages)
    )
    # Report the pure After-Tax Value (strip the 1e-6 tiebreak term)
    objective_value = m.ObjVal
    if objective is Objective.WEALTH_AT_RETIREMENT:
        objective_value -= 1e-6 * value(atv_terminal)
    return Plan(objective=objective, objective_value=objective_value, rows=rows)
