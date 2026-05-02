import pytest
from models import CustomerProfile, ACCOUNT_TYPES
from nodes.lp_optimizer import run_lp_optimizer, compute_roth_ira_limit, IRS_LIMITS


def _profile(**overrides) -> CustomerProfile:
    defaults = dict(
        age=42,
        retirement_age=65,
        annual_income=120_000,
        annual_expenses=85_000,
        retirement_annual_expenses=75_000,
        retirement_years_to_plan=30,
        expected_retirement_income=30_000,
        balances={a: 0.0 for a in ACCOUNT_TYPES},
        employer_match_rate=0.50,
        employer_match_cap=0.06,
        hdhp_enrolled=True,
        current_marginal_tax_rate=0.22,
        assumed_retirement_marginal_tax_rate=0.12,
        filing_status="single",
    )
    defaults.update(overrides)
    return CustomerProfile(**defaults)


def _run(profile: CustomerProfile) -> dict:
    return run_lp_optimizer({"customer_profile": profile})


# --- Core allocation properties ---

def test_solver_returns_allocation_and_wealth():
    result = _run(_profile())
    assert "contribution_allocation" in result
    assert "projected_wealth" in result
    assert "optimization_diagnostics" in result
    assert result["projected_wealth"] > 0


def test_optimizer_returns_gurobi_like_diagnostics():
    result = _run(_profile())
    diagnostics = result["optimization_diagnostics"]

    assert diagnostics.solver == "cvxpy.CLARABEL"
    assert diagnostics.status in ("optimal", "optimal_inaccurate")
    assert diagnostics.objective_value > 0
    assert {v.name for v in diagnostics.variables} == {
        f"{phase}_{account}"
        for phase in ("pre50", "post50")
        for account in ACCOUNT_TYPES
    }


def test_optimizer_diagnostics_identify_binding_constraints():
    result = _run(_profile())
    constraints = {c.name: c for c in result["optimization_diagnostics"].constraints}

    assert constraints["pre50_savings_capacity"].binding
    assert constraints["post50_savings_capacity"].binding
    assert constraints["pre50_savings_capacity"].dual_value is not None
    assert constraints["post50_savings_capacity"].dual_value is not None


def test_budget_not_exceeded():
    p = _profile()
    result = _run(p)
    alloc = result["contribution_allocation"]
    assert sum(alloc.pre50.values()) <= p.savings_capacity + 1e-6
    assert sum(alloc.post50.values()) <= p.savings_capacity + 1e-6


def test_hsa_used_when_hdhp_enrolled():
    # HSA has 1.0 multiplier (triple tax advantage); optimizer should use it.
    # It may not max it when competing with equal-multiplier Roth accounts under a tight budget.
    result = _run(_profile(hdhp_enrolled=True))
    alloc = result["contribution_allocation"]
    assert alloc.pre50["hsa"] > 0
    assert alloc.pre50["hsa"] <= IRS_LIMITS["pre50"]["hsa"] + 1e-6
    assert alloc.post50["hsa"] > 0
    assert alloc.post50["hsa"] <= IRS_LIMITS["post50"]["hsa"] + 1e-6


def test_hsa_zero_when_not_hdhp_enrolled():
    result = _run(_profile(hdhp_enrolled=False))
    alloc = result["contribution_allocation"]
    assert alloc.pre50["hsa"] == pytest.approx(0, abs=1e-6)
    assert alloc.post50["hsa"] == pytest.approx(0, abs=1e-6)


def test_employer_match_captured():
    p = _profile()
    match_cap_dollars = p.employer_match_cap * p.annual_income  # $7,200
    result = _run(p)
    alloc = result["contribution_allocation"]
    pre_401k_total = alloc.pre50["401k"] + alloc.pre50["roth_401k"]
    post_401k_total = alloc.post50["401k"] + alloc.post50["roth_401k"]
    assert pre_401k_total >= match_cap_dollars - 1e-3
    assert post_401k_total >= match_cap_dollars - 1e-3


def test_combined_401k_limit_respected():
    result = _run(_profile())
    alloc = result["contribution_allocation"]
    assert alloc.pre50["401k"] + alloc.pre50["roth_401k"] <= IRS_LIMITS["pre50"]["401k"] + 1e-6
    assert alloc.post50["401k"] + alloc.post50["roth_401k"] <= IRS_LIMITS["post50"]["401k"] + 1e-6


def test_no_negative_contributions():
    result = _run(_profile())
    alloc = result["contribution_allocation"]
    for a in ACCOUNT_TYPES:
        assert alloc.pre50[a] >= -1e-6
        assert alloc.post50[a] >= -1e-6


# --- Phase behaviour ---

def test_already_50_pre50_phase_is_zero():
    result = _run(_profile(age=52))
    alloc = result["contribution_allocation"]
    assert all(v == pytest.approx(0, abs=1e-6) for v in alloc.pre50.values())


def test_post50_catch_up_limits_used_when_age_50():
    result = _run(_profile(age=50))
    alloc = result["contribution_allocation"]
    assert alloc.post50["roth_ira"] <= IRS_LIMITS["post50"]["roth_ira"] + 1e-6
    assert alloc.post50["401k"] + alloc.post50["roth_401k"] <= IRS_LIMITS["post50"]["401k"] + 1e-6


def test_hsa_does_not_get_age_50_catch_up():
    # HSA catch-up starts at 55, which v1's two-phase age-50 allocation cannot model.
    # Keep the post-50 HSA limit at the base family/self-only limit instead of
    # recommending invalid age-50 to age-54 annual contributions.
    assert IRS_LIMITS["post50"]["hsa"] == IRS_LIMITS["pre50"]["hsa"]


# --- Roth IRA phase-out ---

def test_roth_ira_full_limit_below_phase_out():
    limit = compute_roth_ira_limit(100_000, "single", 7_000)
    assert limit == 7_000


def test_roth_ira_zero_above_phase_out():
    limit = compute_roth_ira_limit(170_000, "single", 7_000)
    assert limit == 0.0


def test_roth_ira_tapered_within_phase_out():
    limit = compute_roth_ira_limit(153_500, "single", 7_000)
    assert 0 < limit < 7_000


def test_roth_ira_zero_in_allocation_when_income_above_limit():
    result = _run(_profile(annual_income=200_000, annual_expenses=150_000))
    alloc = result["contribution_allocation"]
    assert alloc.pre50["roth_ira"] == pytest.approx(0, abs=1e-6)
    assert alloc.post50["roth_ira"] == pytest.approx(0, abs=1e-6)


# --- Projected wealth sanity ---

def test_projected_wealth_increases_with_higher_savings():
    low = _run(_profile(annual_income=120_000, annual_expenses=100_000))
    high = _run(_profile(annual_income=120_000, annual_expenses=70_000))
    assert high["projected_wealth"] > low["projected_wealth"]
