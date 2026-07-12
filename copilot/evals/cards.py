"""Fact Cards: repo-mastered ground truth for eval Scenarios (CONTEXT.md, ADR-0004).

A card is both script and answer key: the `facts` prose is everything the
Simulated User may say, and `expected_args` are the exact solve_plan arguments
the agent must produce — unlisted optional fields must be absent from the call.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ..chat import SOLVE_TOOL
from .numbers import extract_numbers, numbers_in

CARDS_DIR = Path(__file__).parent / "cards"

FAMILIES = ("happy_path", "stated_overrides", "infeasible_profile")

_SCHEMA_PROPERTIES = set(SOLVE_TOOL["function"]["parameters"]["properties"])
_SCHEMA_REQUIRED = set(SOLVE_TOOL["function"]["parameters"]["required"])


@dataclass(frozen=True)
class FactCard:
    name: str
    family: str
    facts: str
    expected_args: dict


def load_cards(directory: Path = CARDS_DIR) -> list[FactCard]:
    """Load and validate every Fact Card JSON in `directory`, sorted by name."""
    cards: list[FactCard] = []
    for path in sorted(directory.glob("*.json")):
        card = FactCard(**json.loads(path.read_text()))
        if card.family not in FAMILIES:
            raise ValueError(f"{path.name}: unknown family {card.family!r}")
        missing = _SCHEMA_REQUIRED - set(card.expected_args)
        unknown = set(card.expected_args) - _SCHEMA_PROPERTIES
        if missing or unknown:
            raise ValueError(
                f"{path.name}: expected_args missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        cards.append(card)
    if not cards:
        raise ValueError(f"no Fact Cards found in {directory}")
    return cards


def card_numbers(card: FactCard) -> tuple[float, ...]:
    """Every number the card states — the Simulated User's full permitted set."""
    values = list(numbers_in(card.expected_args))
    values.extend(token.value for token in extract_numbers(card.facts))
    return tuple(values)
