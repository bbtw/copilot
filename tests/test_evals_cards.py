import pytest

from copilot.evals.cards import card_numbers, load_cards
from copilot.model import solve
from copilot.plan import Infeasible, Plan
from copilot.profile import from_solve_args

CARDS = load_cards()


def test_all_families_represented() -> None:
    assert len(CARDS) >= 6
    assert {card.family for card in CARDS} == {
        "happy_path",
        "stated_overrides",
        "infeasible_profile",
    }
    names = [card.name for card in CARDS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_expected_args_build_a_valid_profile(card) -> None:
    profile, objective = from_solve_args(card.expected_args)
    assert profile is not None and objective is not None


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card_solves_as_its_family_claims(card) -> None:
    profile, objective = from_solve_args(card.expected_args)
    result = solve(profile, objective)
    if card.family == "infeasible_profile":
        assert isinstance(result, Infeasible)
    else:
        assert isinstance(result, Plan)


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_facts_numbers_cover_expected_args(card) -> None:
    # Sanity: every arg number appears among the card's numbers, so a sim user
    # quoting the sheet can never be voided for stating a required fact
    values = set(card_numbers(card))
    for key, arg in card.expected_args.items():
        if isinstance(arg, (int, float)) and not isinstance(arg, bool):
            assert float(arg) in values, f"{key}={arg} not stated in facts/args"
