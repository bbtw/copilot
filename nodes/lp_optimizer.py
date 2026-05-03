import cvxpy as cp
from models import (
    ACCOUNT_TYPES,
    ContributionAllocation,
    OptimizationConstraintDiagnostic,
    OptimizationDiagnostics,
    OptimizationVariableDiagnostic,
)
from state import RetirementPlanState

EXPECTED_RETURN = 0.07

# 2024 IRS limits
IRS_LIMITS: dict[str, dict[str, float]] = {
    "pre50": {
        "401k": 23_000,
        "roth_401k": 23_000,   # shared with 401k — enforced via combined constraint
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
        "hsa": 4_150,
        "taxable_brokerage": float("inf"),
    },
}

# After-tax multiplier applied to terminal balance per account type
def after_tax_multiplier(acct: str, retirement_tax_rate: float) -> float:
    if acct in ("roth_401k", "roth_ira", "hsa"):
        return 1.0
    elif acct in ("401k", "traditional_ira"):
        return 1.0 - retirement_tax_rate
    else:  # taxable_brokerage
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


def run_lp_optimizer(state: RetirementPlanState) -> dict:
    p = state.customer_profile
    r = EXPECTED_RETURN
    tax_out = p.assumed_retirement_marginal_tax_rate

    pre50_yrs = p.pre50_years
    post50_yrs = p.post50_years

    # FV annuity factors: contributions compound for remaining years after each phase
    # Pre-50 contributions compound for (pre50_yrs + post50_yrs) years total...
    # but each year's contribution compounds differently. Using end-of-phase lump sum:
    # pre50 annual contrib → FV at end of pre50 phase → then compounds post50_yrs more years
    fv_pre50 = fv_annuity(r, pre50_yrs) * (1 + r) ** post50_yrs
    fv_post50 = fv_annuity(r, post50_yrs)

    # Existing balance compounds full years_to_retirement
    fv_existing = (1 + r) ** p.years_to_retirement

    n = len(ACCOUNT_TYPES)
    idx = {a: i for i, a in enumerate(ACCOUNT_TYPES)}

    # Decision variables. Bounds are explicit constraints so we can expose
    # lower-bound duals, similar to Gurobi's reduced-cost diagnostics.
    x_pre = cp.Variable(n, name="x_pre")    # pre-50 annual contributions
    x_post = cp.Variable(n, name="x_post")  # post-50 annual contributions

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

    # Objective: maximize after-tax projected wealth
    existing_wealth = sum(
        p.balances.get(a, 0) * fv_existing * multipliers[idx[a]]
        for a in ACCOUNT_TYPES
    )
    contrib_wealth = (
        cp.sum(cp.multiply(multipliers, x_pre * fv_pre50)) +
        cp.sum(cp.multiply(multipliers, x_post * fv_post50))
    )
    objective = cp.Maximize(contrib_wealth)

    constraints = []
    named_constraints = []

    def add_constraint(name, category, phase, constraint, slack_fn):
        constraints.append(constraint)
        named_constraints.append(
            {
                "name": name,
                "category": category,
                "phase": phase,
                "constraint": constraint,
                "slack_fn": slack_fn,
            }
        )

    lower_bound_constraints: dict[tuple[str, str], cp.constraints.constraint.Constraint] = {}

    for i, a in enumerate(ACCOUNT_TYPES):
        c_pre = x_pre[i] >= 0
        lower_bound_constraints[("pre50", a)] = c_pre
        add_constraint(
            f"pre50_{a}_lower_bound",
            "variable_lower_bound",
            "pre50",
            c_pre,
            lambda i=i: x_pre.value[i],
        )

        c_post = x_post[i] >= 0
        lower_bound_constraints[("post50", a)] = c_post
        add_constraint(
            f"post50_{a}_lower_bound",
            "variable_lower_bound",
            "post50",
            c_post,
            lambda i=i: x_post.value[i],
        )

    # IRS per-account limits
    for i, a in enumerate(ACCOUNT_TYPES):
        if limits_pre[i] < float("inf"):
            add_constraint(
                f"pre50_{a}_irs_limit",
                "irs_limit",
                "pre50",
                x_pre[i] <= limits_pre[i],
                lambda i=i, limit=limits_pre[i]: limit - x_pre.value[i],
            )
        if limits_post[i] < float("inf"):
            add_constraint(
                f"post50_{a}_irs_limit",
                "irs_limit",
                "post50",
                x_post[i] <= limits_post[i],
                lambda i=i, limit=limits_post[i]: limit - x_post.value[i],
            )

    # Combined 401k + Roth 401k elective deferral limit
    i401k, iroth401k = idx["401k"], idx["roth_401k"]
    add_constraint(
        "pre50_combined_401k_limit",
        "combined_401k_limit",
        "pre50",
        x_pre[i401k] + x_pre[iroth401k] <= IRS_LIMITS["pre50"]["401k"],
        lambda: IRS_LIMITS["pre50"]["401k"] - x_pre.value[i401k] - x_pre.value[iroth401k],
    )
    add_constraint(
        "post50_combined_401k_limit",
        "combined_401k_limit",
        "post50",
        x_post[i401k] + x_post[iroth401k] <= IRS_LIMITS["post50"]["401k"],
        lambda: IRS_LIMITS["post50"]["401k"] - x_post.value[i401k] - x_post.value[iroth401k],
    )

    # HSA: only available if HDHP enrolled
    if not p.hdhp_enrolled:
        add_constraint(
            "pre50_hsa_unavailable",
            "eligibility",
            "pre50",
            x_pre[idx["hsa"]] == 0,
            lambda: abs(x_pre.value[idx["hsa"]]),
        )
        add_constraint(
            "post50_hsa_unavailable",
            "eligibility",
            "post50",
            x_post[idx["hsa"]] == 0,
            lambda: abs(x_post.value[idx["hsa"]]),
        )

    # Budget constraint: total contributions ≤ SavingsCapacity
    # Employer match doesn't draw from SavingsCapacity
    add_constraint(
        "pre50_savings_capacity",
        "budget",
        "pre50",
        cp.sum(x_pre) <= p.savings_capacity,
        lambda: p.savings_capacity - sum(x_pre.value),
    )
    add_constraint(
        "post50_savings_capacity",
        "budget",
        "post50",
        cp.sum(x_post) <= p.savings_capacity,
        lambda: p.savings_capacity - sum(x_post.value),
    )

    # No pre-50 phase for customers already 50+
    if pre50_yrs == 0:
        add_constraint(
            "pre50_phase_not_applicable",
            "phase_applicability",
            "pre50",
            cp.sum(x_pre) == 0,
            lambda: abs(sum(x_pre.value)),
        )

    # Employer match: matched = match_rate * min(401k + roth_401k, match_cap * income)
    # Add match value to objective via auxiliary variable
    match_cap_dollars = p.employer_match_cap * p.annual_income
    m_pre = cp.Variable(name="m_pre")
    m_post = cp.Variable(name="m_post")
    add_constraint(
        "pre50_match_lower_bound",
        "variable_lower_bound",
        "pre50",
        m_pre >= 0,
        lambda: m_pre.value,
    )
    add_constraint(
        "post50_match_lower_bound",
        "variable_lower_bound",
        "post50",
        m_post >= 0,
        lambda: m_post.value,
    )
    # We use auxiliary variables `m_pre` and `m_post` to represent the *eligible* contribution amount 
    # that the employer will match. This amount is bounded by two distinct conditions:
    # 1. It cannot exceed the actual employee contributions made to 401k and Roth 401k.
    add_constraint(
        "pre50_match_cannot_exceed_401k_contributions",
        "employer_match",
        "pre50",
        m_pre <= x_pre[i401k] + x_pre[iroth401k],
        lambda: x_pre.value[i401k] + x_pre.value[iroth401k] - m_pre.value,
    )
    # 2. It cannot exceed the employer's maximum match cap (calculated as a % of annual income).
    add_constraint(
        "pre50_match_cap",
        "employer_match",
        "pre50",
        m_pre <= match_cap_dollars,
        lambda: match_cap_dollars - m_pre.value,
    )
    add_constraint(
        "post50_match_cannot_exceed_401k_contributions",
        "employer_match",
        "post50",
        m_post <= x_post[i401k] + x_post[iroth401k],
        lambda: x_post.value[i401k] + x_post.value[iroth401k] - m_post.value,
    )
    add_constraint(
        "post50_match_cap",
        "employer_match",
        "post50",
        m_post <= match_cap_dollars,
        lambda: match_cap_dollars - m_post.value,
    )

    # Matched dollars go into 401k (pre-tax), add their compounded after-tax value to objective
    match_tax_mult = after_tax_multiplier("401k", tax_out)
    match_wealth = (
        p.employer_match_rate * m_pre * fv_pre50 * match_tax_mult +
        p.employer_match_rate * m_post * fv_post50 * match_tax_mult
    )
    objective = cp.Maximize(contrib_wealth + match_wealth)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP solver failed: {prob.status}")

    if pre50_yrs == 0:
        pre_alloc = {a: 0.0 for a in ACCOUNT_TYPES}
    else:
        pre_alloc = {a: float(max(x_pre.value[idx[a]], 0)) for a in ACCOUNT_TYPES}
    post_alloc = {a: float(max(x_post.value[idx[a]], 0)) for a in ACCOUNT_TYPES}

    projected_wealth = float(prob.value) + existing_wealth

    # --- Post-Optimization Analysis ---
    # Extract detailed diagnostics from the solver's result.
    # We collect the dual values (often called shadow prices or reduced costs).
    # These tell us the marginal value of relaxing a constraint—i.e., how much the 
    # total projected wealth would increase if a constraint limit were raised by $1.
    variable_diagnostics = []
    for phase, values, fv_factor in (
        ("pre50", x_pre.value, fv_pre50),
        ("post50", x_post.value, fv_post50),
    ):
        for a in ACCOUNT_TYPES:
            lower_bound_dual = _dual_value(lower_bound_constraints[(phase, a)])
            value = float(max(values[idx[a]], 0))
            variable_diagnostics.append(
                OptimizationVariableDiagnostic(
                    name=f"{phase}_{a}",
                    phase=phase,
                    account=a,
                    value=value,
                    objective_coefficient=float(multipliers[idx[a]] * fv_factor),
                    reduced_cost=None if lower_bound_dual is None else abs(lower_bound_dual),
                    at_lower_bound=value <= 1e-5,
                )
            )

    constraint_diagnostics = []
    for item in named_constraints:
        slack = float(item["slack_fn"]())
        constraint_diagnostics.append(
            OptimizationConstraintDiagnostic(
                name=item["name"],
                category=item["category"],
                phase=item["phase"],
                slack=slack,
                dual_value=_dual_value(item["constraint"]),
                binding=abs(slack) <= 1e-4,
            )
        )

    diagnostics = OptimizationDiagnostics(
        solver="cvxpy.CLARABEL",
        status=prob.status,
        objective_value=float(prob.value),
        variables=variable_diagnostics,
        constraints=constraint_diagnostics,
    )

    return {
        "contribution_allocation": ContributionAllocation(pre50=pre_alloc, post50=post_alloc),
        "projected_wealth": projected_wealth,
        "optimization_diagnostics": diagnostics,
    }
