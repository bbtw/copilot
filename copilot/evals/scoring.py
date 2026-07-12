"""The two deterministic scorers — Profile Fidelity and Number Faithfulness —
plus the Simulated User audit that shares the faithfulness extractor (ADR-0004)."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..taxdata import (
    DEFAULT_CAPITAL_GAINS_RATE,
    DEFAULT_CONTRIBUTION_LIMIT,
    DEFAULT_REAL_RETURN,
    EARLY_WITHDRAWAL_AGE,
    RMD_START_AGE,
    SS_TAXABLE_FRACTION,
    FilingStatus,
    ordinary_segments,
)
from .cards import FactCard, card_numbers
from .numbers import extract_numbers, matches_any, numbers_in


@dataclass(frozen=True)
class Score:
    passed: bool
    reasons: tuple[str, ...] = ()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalized_streams(streams: object) -> list[tuple[str, float, int]] | None:
    if not isinstance(streams, list):
        return None
    try:
        return sorted(
            (str(s["kind"]), float(s["annual_amount"]), int(s["start_age"])) for s in streams
        )
    except (KeyError, TypeError, ValueError):
        return None


def _values_equal(key: str, want: object, got: object) -> bool:
    if key == "income_streams":
        normalized = _normalized_streams(got)
        return normalized is not None and normalized == _normalized_streams(want)
    if _is_number(want):
        return _is_number(got) and math.isclose(
            float(want), float(got), rel_tol=1e-9, abs_tol=1e-9
        )
    return want == got


def profile_fidelity(solve_args: dict | None, card: FactCard) -> Score:
    """Do the raw solve_plan arguments match the Fact Card exactly — stated
    fields verbatim, unstated optional fields strictly absent (CONTEXT.md)."""
    if solve_args is None:
        return Score(
            False, ("no-solve: no solve_plan call with parseable arguments before the turn cap",)
        )
    reasons: list[str] = []
    for key, want in card.expected_args.items():
        if key not in solve_args:
            reasons.append(f"{key}: stated on the card but missing from the call")
        elif not _values_equal(key, want, solve_args[key]):
            reasons.append(f"{key}: expected {want!r}, got {solve_args[key]!r}")
    for key in solve_args:
        if key not in card.expected_args:
            reasons.append(f"{key}: passed but not stated on the card (must be absent)")
    return Score(not reasons, tuple(reasons))


def _taxdata_sources() -> tuple[float, ...]:
    values = [
        DEFAULT_REAL_RETURN,
        DEFAULT_CAPITAL_GAINS_RATE,
        DEFAULT_CONTRIBUTION_LIMIT,
        SS_TAXABLE_FRACTION,
        float(EARLY_WITHDRAWAL_AGE),
        float(RMD_START_AGE),
    ]
    for status in FilingStatus:
        segments = ordinary_segments(status)
        values.append(segments[0][0])  # standard deduction leads as the 0%-rate segment
        upper = 0.0
        for width, rate in segments[1:]:
            values.append(rate)
            if width != float("inf"):
                upper += width
                values.append(upper)
    return tuple(values)


_TAXDATA_SOURCES = _taxdata_sources()


def faithfulness_sources(card: FactCard, solve_result: dict | None) -> tuple[float, ...]:
    """The allowed sources for agent prose: the solve_plan result (numbers in an
    infeasibility reason or error message included), the Fact Card, every age
    within the plan, and the taxdata defaults."""
    values = list(card_numbers(card))
    first_age = int(card.expected_args["current_age"])
    last_age = int(card.expected_args["horizon_age"])
    values.extend(float(age) for age in range(first_age, last_age + 1))
    values.extend(_TAXDATA_SOURCES)
    if solve_result is not None:
        values.extend(numbers_in(solve_result))
    return tuple(values)


def number_faithfulness(agent_texts: Sequence[str], sources: tuple[float, ...]) -> Score:
    """Does every number in the agent's prose match a source value, exactly or
    rounded to the displayed significant digits (CONTEXT.md)."""
    reasons = tuple(
        f"'{token.text}' (agent turn {turn}) matches no allowed source"
        for turn, text in enumerate(agent_texts, start=1)
        for token in extract_numbers(text)
        if not matches_any(token, sources)
    )
    return Score(not reasons, reasons)


def audit_sim(sim_texts: Sequence[str], card: FactCard) -> Score:
    """Void check: every number the Simulated User stated must be on its Fact Card."""
    sources = card_numbers(card)
    reasons = tuple(
        f"'{token.text}' (sim turn {turn}) is not on the Fact Card"
        for turn, text in enumerate(sim_texts, start=1)
        for token in extract_numbers(text)
        if not matches_any(token, sources)
    )
    return Score(not reasons, reasons)
