from dataclasses import replace

import pytest

from copilot.model import solve
from copilot.plan import Infeasible, Objective, Plan
from copilot.profile import IncomeKind, IncomeStream, Profile
from copilot.taxdata import FilingStatus, rmd_divisor, tax_on_ordinary_income

TOL = 1e-3


def baseline_profile(**overrides) -> Profile:
    profile = Profile(
        current_age=40,
        retirement_age=65,
        horizon_age=90,
        filing_status=FilingStatus.SINGLE,
        gross_income=150_000,
        savings_capacity=30_000,
        spending_need=80_000,
        balance_traditional=200_000,
        balance_roth=50_000,
        balance_taxable=100_000,
        income_streams=(IncomeStream(IncomeKind.SOCIAL_SECURITY, 30_000, 67),),
    )
    return replace(profile, **overrides)


def gi_cash(profile: Profile, age: int) -> float:
    return sum(s.annual_amount for s in profile.income_streams if s.start_age <= age)


def test_baseline_solves_and_meets_spending() -> None:
    profile = baseline_profile()
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(plan, Plan)
    for row in plan.rows:
        assert row.balance_traditional >= -TOL
        assert row.balance_roth >= -TOL
        assert row.balance_taxable >= -TOL
        if row.phase == "decumulation":
            withdrawals = (
                row.withdraw_traditional + row.withdraw_roth + row.withdraw_taxable
            )
            assert withdrawals + gi_cash(profile, row.age) - row.tax >= (
                profile.spending_need - TOL
            )


def test_contribution_limit_respected() -> None:
    profile = baseline_profile()
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    for row in plan.rows:
        if row.phase == "accumulation":
            assert row.contrib_traditional + row.contrib_roth <= (
                profile.contribution_limit + TOL
            )


def test_no_retirement_account_withdrawals_before_60() -> None:
    profile = baseline_profile(
        current_age=50,
        retirement_age=55,
        balance_taxable=1_500_000,
    )
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(plan, Plan)
    for row in plan.rows:
        if row.phase == "decumulation" and row.age < 60:
            assert row.withdraw_traditional <= TOL
            assert row.withdraw_roth <= TOL


def test_rmd_floor_satisfied() -> None:
    profile = baseline_profile(balance_traditional=2_000_000, spending_need=60_000)
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(plan, Plan)
    previous_traditional = profile.balance_traditional
    for row in plan.rows:
        if row.age >= 73:
            assert row.withdraw_traditional >= (
                previous_traditional / rmd_divisor(row.age) - TOL
            )
        previous_traditional = row.balance_traditional


def test_high_earner_low_spender_favors_traditional() -> None:
    profile = baseline_profile(
        gross_income=300_000,
        savings_capacity=40_000,
        spending_need=60_000,
        income_streams=(IncomeStream(IncomeKind.SOCIAL_SECURITY, 30_000, 70),),
    )
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(plan, Plan)
    accumulation = [r for r in plan.rows if r.phase == "accumulation"]
    total_traditional = sum(r.contrib_traditional for r in accumulation)
    total_roth = sum(r.contrib_roth for r in accumulation)
    assert total_traditional > total_roth


def test_conversions_appear_in_low_bracket_gap_years() -> None:
    profile = baseline_profile(
        current_age=60,
        retirement_age=60,
        gross_income=0,
        savings_capacity=0,
        spending_need=60_000,
        balance_traditional=2_000_000,
        balance_roth=0,
        balance_taxable=800_000,
        income_streams=(IncomeStream(IncomeKind.SOCIAL_SECURITY, 40_000, 70),),
    )
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(plan, Plan)
    gap_conversions = sum(r.conversion for r in plan.rows if 60 <= r.age < 70)
    assert gap_conversions > 1_000


def test_objective_is_after_tax_value_not_raw_balance() -> None:
    profile = baseline_profile()
    plan = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(plan, Plan)
    last = plan.rows[-1]
    expected = (
        last.balance_roth
        + last.balance_taxable
        + last.balance_traditional
        - tax_on_ordinary_income(last.balance_traditional, profile.filing_status)
    )
    assert plan.objective_value == pytest.approx(expected, rel=1e-4)


def test_impossible_spending_is_infeasible_with_reason() -> None:
    profile = baseline_profile(
        current_age=60,
        retirement_age=60,
        gross_income=0,
        savings_capacity=0,
        spending_need=500_000,
        balance_traditional=50_000,
        balance_roth=25_000,
        balance_taxable=25_000,
        income_streams=(),
    )
    result = solve(profile, Objective.TERMINAL_WEALTH)
    assert isinstance(result, Infeasible)
    assert "infeasible" in result.reason.lower()


def test_full_lifetime_model_fits_free_license() -> None:
    profile = baseline_profile(current_age=25, retirement_age=65, horizon_age=95)
    plan = solve(profile, Objective.WEALTH_AT_RETIREMENT)
    # solve() raises ModelTooLargeError past the 2,000 var/constraint trial cap
    assert isinstance(plan, Plan)
