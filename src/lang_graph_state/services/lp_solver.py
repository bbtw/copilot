from typing import Any

import cvxpy as cp

from lang_graph_state.domain.models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    CustomerProfile,
    OptimizationConstraintDiagnostic,
    OptimizationDiagnostics,
    OptimizationVariableDiagnostic,
)
from lang_graph_state.instrumentation.timing import timed

EXPECTED_RETURN = 0.07

# 2024 IRS contribution limits
IRS_LIMITS: dict[str, dict[str, float]] = {
    "pre50": {
        "401k": 23_000,
        "roth_401k": 23_000,  # shared with 401k — enforced via combined constraint
        "traditional_ira": 7_000,
        "roth_ira": 7_000,
        "hsa": 4_150,
        "taxable_brokerage": float("inf"),
    },
    "post50": {
        "401k": 30_500,
        "roth_401k": 30_500,
        "traditional_ira": 8_000,
        "roth_ira": 8_000,
        "hsa": 4_150,  # HSA catch-up begins at 55; v1 uses base limit in both phases
        "taxable_brokerage": float("inf"),
    },
}


def after_tax_multiplier(acct: str, retirement_tax_rate: float) -> float:
    if acct in ("roth_401k", "roth_ira", "hsa"):
        return 1.0
    return 1.0 - retirement_tax_rate


def fv_annuity(r: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return ((1 + r) ** n - 1) / r


def compute_roth_ira_limit(annual_income: float, filing_status: str, base_limit: float) -> float:
    """Apply Roth IRA income phase-out taper (2024 thresholds)."""
    if filing_status == "single":
        phase_out_start, phase_out_end = 146_000, 161_000
    elif filing_status == "married_filing_jointly":
        phase_out_start, phase_out_end = 230_000, 240_000
    else:
        phase_out_start, phase_out_end = 0, 10_000

    if annual_income <= phase_out_start:
        return base_limit
    if annual_income >= phase_out_end:
        return 0.0
    taper = 1.0 - (annual_income - phase_out_start) / (phase_out_end - phase_out_start)
    return round(taper * base_limit / 10) * 10  # IRS rounds to nearest $10


def _dual_value(constraint: cp.constraints.constraint.Constraint) -> float | None:
    if constraint.dual_value is None:
        return None
    try:
        return float(constraint.dual_value)
    except TypeError:
        return float(constraint.dual_value.item())


@timed(
    "lp_solve",
    finish_attrs=lambda result: {
        "projected_wealth": f"{result['projected_wealth']:.2f}",
        "binding_constraints": sum(1 for c in result["optimization_diagnostics"].constraints if c.binding),
    },
)
def solve_lp(profile: CustomerProfile) -> dict[str, Any]:
    """
    Solve the retirement contribution LP for the given profile.
    Returns contribution_allocation, projected_wealth, and optimization_diagnostics.
    """
    p = profile
    r = EXPECTED_RETURN
    tax_out = p.assumed_retirement_marginal_tax_rate

    pre50_yrs = p.pre50_years
    post50_yrs = p.post50_years

    # Pre-50 annual contributions compound through the pre-50 phase then for the full post-50 period.
    fv_pre50 = fv_annuity(r, pre50_yrs) * (1 + r) ** post50_yrs
    fv_post50 = fv_annuity(r, post50_yrs)
    fv_existing = (1 + r) ** p.years_to_retirement

    n = len(ACCOUNT_TYPES)
    idx = {a: i for i, a in enumerate(ACCOUNT_TYPES)}

    # Explicit bound constraints expose lower-bound duals (reduced-cost diagnostics).
    x_pre = cp.Variable(n, name="x_pre")
    x_post = cp.Variable(n, name="x_post")

    limits_pre = [
        IRS_LIMITS["pre50"][a] if a != "roth_ira"
        else compute_roth_ira_limit(p.annual_income, p.filing_status, IRS_LIMITS["pre50"]["roth_ira"])
        for a in ACCOUNT_TYPES
    ]
    limits_post = [
        IRS_LIMITS["post50"][a] if a != "roth_ira"
        else compute_roth_ira_limit(p.annual_income, p.filing_status, IRS_LIMITS["post50"]["roth_ira"])
        for a in ACCOUNT_TYPES
    ]

    multipliers = [after_tax_multiplier(a, tax_out) for a in ACCOUNT_TYPES]

    existing_wealth = sum(
        p.balances.get(a, 0) * fv_existing * multipliers[idx[a]]
        for a in ACCOUNT_TYPES
    )
    contrib_wealth = (
        cp.sum(cp.multiply(multipliers, x_pre * fv_pre50)) +
        cp.sum(cp.multiply(multipliers, x_post * fv_post50))
    )

    constraints: list = []
    named_constraints: list[dict] = []

    def add(name: str, category: str, phase: str, constraint, slack_fn):
        constraints.append(constraint)
        named_constraints.append({"name": name, "category": category, "phase": phase,
                                   "constraint": constraint, "slack_fn": slack_fn})

    lower_bound_constraints: dict[tuple[str, str], Any] = {}

    for i, a in enumerate(ACCOUNT_TYPES):
        c_pre = x_pre[i] >= 0
        lower_bound_constraints[("pre50", a)] = c_pre
        add(f"pre50_{a}_lower_bound", "variable_lower_bound", "pre50", c_pre, lambda i=i: x_pre.value[i])

        c_post = x_post[i] >= 0
        lower_bound_constraints[("post50", a)] = c_post
        add(f"post50_{a}_lower_bound", "variable_lower_bound", "post50", c_post, lambda i=i: x_post.value[i])

    for i, a in enumerate(ACCOUNT_TYPES):
        if limits_pre[i] < float("inf"):
            add(f"pre50_{a}_irs_limit", "irs_limit", "pre50", x_pre[i] <= limits_pre[i],
                lambda i=i, lim=limits_pre[i]: lim - x_pre.value[i])
        if limits_post[i] < float("inf"):
            add(f"post50_{a}_irs_limit", "irs_limit", "post50", x_post[i] <= limits_post[i],
                lambda i=i, lim=limits_post[i]: lim - x_post.value[i])

    i401k, iroth401k = idx["401k"], idx["roth_401k"]
    add("pre50_combined_401k_limit", "combined_401k_limit", "pre50",
        x_pre[i401k] + x_pre[iroth401k] <= IRS_LIMITS["pre50"]["401k"],
        lambda: IRS_LIMITS["pre50"]["401k"] - x_pre.value[i401k] - x_pre.value[iroth401k])
    add("post50_combined_401k_limit", "combined_401k_limit", "post50",
        x_post[i401k] + x_post[iroth401k] <= IRS_LIMITS["post50"]["401k"],
        lambda: IRS_LIMITS["post50"]["401k"] - x_post.value[i401k] - x_post.value[iroth401k])

    if not p.hdhp_enrolled:
        add("pre50_hsa_unavailable", "eligibility", "pre50", x_pre[idx["hsa"]] == 0,
            lambda: abs(x_pre.value[idx["hsa"]]))
        add("post50_hsa_unavailable", "eligibility", "post50", x_post[idx["hsa"]] == 0,
            lambda: abs(x_post.value[idx["hsa"]]))

    add("pre50_savings_capacity", "budget", "pre50", cp.sum(x_pre) <= p.savings_capacity,
        lambda: p.savings_capacity - sum(x_pre.value))
    add("post50_savings_capacity", "budget", "post50", cp.sum(x_post) <= p.savings_capacity,
        lambda: p.savings_capacity - sum(x_post.value))

    if pre50_yrs == 0:
        add("pre50_phase_not_applicable", "phase_applicability", "pre50", cp.sum(x_pre) == 0,
            lambda: abs(sum(x_pre.value)))

    # Employer match: matched = match_rate * min(401k + roth_401k contributions, match_cap * income)
    # Modeled via auxiliary variable m so the piecewise-linear min stays LP-representable.
    match_cap_dollars = p.employer_match_cap * p.annual_income
    m_pre = cp.Variable(name="m_pre")
    m_post = cp.Variable(name="m_post")
    add("pre50_match_lower_bound", "variable_lower_bound", "pre50", m_pre >= 0, lambda: m_pre.value)
    add("post50_match_lower_bound", "variable_lower_bound", "post50", m_post >= 0, lambda: m_post.value)
    add("pre50_match_cannot_exceed_401k_contributions", "employer_match", "pre50",
        m_pre <= x_pre[i401k] + x_pre[iroth401k],
        lambda: x_pre.value[i401k] + x_pre.value[iroth401k] - m_pre.value)
    add("pre50_match_cap", "employer_match", "pre50", m_pre <= match_cap_dollars,
        lambda: match_cap_dollars - m_pre.value)
    add("post50_match_cannot_exceed_401k_contributions", "employer_match", "post50",
        m_post <= x_post[i401k] + x_post[iroth401k],
        lambda: x_post.value[i401k] + x_post.value[iroth401k] - m_post.value)
    add("post50_match_cap", "employer_match", "post50", m_post <= match_cap_dollars,
        lambda: match_cap_dollars - m_post.value)

    match_tax_mult = after_tax_multiplier("401k", tax_out)
    match_wealth = (
        p.employer_match_rate * m_pre * fv_pre50 * match_tax_mult +
        p.employer_match_rate * m_post * fv_post50 * match_tax_mult
    )

    prob = cp.Problem(cp.Maximize(contrib_wealth + match_wealth), constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP solver failed: {prob.status}")

    pre_alloc = {a: 0.0 for a in ACCOUNT_TYPES} if pre50_yrs == 0 else \
                {a: float(max(x_pre.value[idx[a]], 0)) for a in ACCOUNT_TYPES}
    post_alloc = {a: float(max(x_post.value[idx[a]], 0)) for a in ACCOUNT_TYPES}

    variable_diagnostics = []
    for phase, values, fv_factor in (("pre50", x_pre.value, fv_pre50), ("post50", x_post.value, fv_post50)):
        for a in ACCOUNT_TYPES:
            lower_bound_dual = _dual_value(lower_bound_constraints[(phase, a)])
            value = float(max(values[idx[a]], 0))
            variable_diagnostics.append(OptimizationVariableDiagnostic(
                name=f"{phase}_{a}", phase=phase, account=a, value=value,
                objective_coefficient=float(multipliers[idx[a]] * fv_factor),
                reduced_cost=None if lower_bound_dual is None else abs(lower_bound_dual),
                at_lower_bound=value <= 1e-5,
            ))

    constraint_diagnostics = [
        OptimizationConstraintDiagnostic(
            name=item["name"], category=item["category"], phase=item["phase"],
            slack=float(item["slack_fn"]()),
            dual_value=_dual_value(item["constraint"]),
            binding=abs(float(item["slack_fn"]())) <= 1e-4,
        )
        for item in named_constraints
    ]

    return {
        "contribution_allocation": ContributionAllocation(pre50=pre_alloc, post50=post_alloc),
        "projected_wealth": float(prob.value) + existing_wealth,
        "optimization_diagnostics": OptimizationDiagnostics(
            solver="cvxpy.CLARABEL", status=prob.status,
            objective_value=float(prob.value),
            variables=variable_diagnostics, constraints=constraint_diagnostics,
        ),
    }
