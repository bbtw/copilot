from copilot.evals.cards import FactCard
from copilot.evals.scoring import (
    audit_sim,
    faithfulness_sources,
    number_faithfulness,
    profile_fidelity,
)

CARD = FactCard(
    name="test-card",
    family="happy_path",
    facts=(
        "Goal: maximize terminal wealth to age 90.\n"
        "- Age 40, single, retiring at 65.\n"
        "- Income $150,000; saves $30,000; spends $80,000 retired.\n"
        "- $200,000 traditional, $50,000 Roth, $100,000 taxable.\n"
        "- Social Security $30,000 starting at 67."
    ),
    expected_args={
        "current_age": 40,
        "retirement_age": 65,
        "horizon_age": 90,
        "filing_status": "single",
        "gross_income": 150000,
        "savings_capacity": 30000,
        "spending_need": 80000,
        "balance_traditional": 200000,
        "balance_roth": 50000,
        "balance_taxable": 100000,
        "income_streams": [
            {"kind": "social_security", "annual_amount": 30000, "start_age": 67}
        ],
        "objective": "terminal_wealth",
    },
)


def test_fidelity_exact_args_pass() -> None:
    assert profile_fidelity(dict(CARD.expected_args), CARD).passed


def test_fidelity_explicit_unstated_default_fails() -> None:
    args = dict(CARD.expected_args, expected_return=0.05)
    score = profile_fidelity(args, CARD)
    assert not score.passed
    assert any("expected_return" in reason for reason in score.reasons)


def test_fidelity_missing_stated_field_fails() -> None:
    args = dict(CARD.expected_args)
    del args["spending_need"]
    assert not profile_fidelity(args, CARD).passed


def test_fidelity_wrong_value_fails() -> None:
    args = dict(CARD.expected_args, retirement_age=66)
    assert not profile_fidelity(args, CARD).passed


def test_fidelity_stream_order_is_irrelevant() -> None:
    streams = [
        {"kind": "pension", "annual_amount": 10000, "start_age": 65},
        {"kind": "social_security", "annual_amount": 30000, "start_age": 67},
    ]
    card = FactCard(
        name="two-streams",
        family="happy_path",
        facts=CARD.facts,
        expected_args=dict(CARD.expected_args, income_streams=streams),
    )
    args = dict(card.expected_args, income_streams=list(reversed(streams)))
    assert profile_fidelity(args, card).passed


def test_fidelity_no_solve_fails() -> None:
    score = profile_fidelity(None, CARD)
    assert not score.passed
    assert "no-solve" in score.reasons[0]


def test_faithfulness_rounded_solve_number_passes() -> None:
    sources = faithfulness_sources(CARD, {"objective_value_after_tax": 1_203_456})
    score = number_faithfulness(["Your after-tax value at 90 is about $1.2M."], sources)
    assert score.passed


def test_faithfulness_derived_number_fails() -> None:
    sources = faithfulness_sources(CARD, {"conversions": {"total": 320000}})
    score = number_faithfulness(["You convert $320,000 total — about $40k a year."], sources)
    assert not score.passed
    assert len(score.reasons) == 1


def test_faithfulness_defaults_allowed_without_solve() -> None:
    sources = faithfulness_sources(CARD, None)
    score = number_faithfulness(
        ["I'll assume a 5% real return, 15% capital gains, and a $24,500 limit."], sources
    )
    assert score.passed


def test_faithfulness_infeasibility_reason_is_a_source() -> None:
    result = {"infeasible": "taxable cannot bridge until accounts unlock at age 60"}
    sources = faithfulness_sources(CARD, result)
    score = number_faithfulness(["Retirement accounts unlock at 60, so this fails."], sources)
    assert score.passed


def test_audit_sim_on_card_passes() -> None:
    texts = ["Hi! I'm 40, single, making $150k. I want to retire at 65."]
    assert audit_sim(texts, CARD).passed


def test_audit_sim_off_card_number_voids() -> None:
    score = audit_sim(["I make $150k and my rent is $2,000."], CARD)
    assert not score.passed
    assert "$2,000" in score.reasons[0]
