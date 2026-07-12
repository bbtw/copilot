"""Deterministic number extraction and significant-digits matching (ADR-0004).

The one matching rule (CONTEXT.md, Number Faithfulness): a displayed number
matches a source value if it equals the source rounded to the significant
digits displayed — "$1.2M" matches 1,203,456; "$1.25M" does not. Derived
arithmetic matches nothing, because a derived figure is not a source value.
"""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

# A prose number: optional $, digits with commas/decimal, optional k/M/million/
# thousand scale, optional %. The lookbehind keeps us out of identifiers.
_TOKEN_RE = re.compile(
    r"(?<![\w.])"
    r"\$?(?P<digits>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<scale>million|thousand|[km]\b))?"
    r"(?P<percent>\s?%)?",
    re.IGNORECASE,
)

_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}

# Markdown enumerators ("1. First, ...") are list syntax, not stated figures
_ENUMERATOR_RE = re.compile(r"^\s{0,8}\d+\.\s", re.MULTILINE)


@dataclass(frozen=True)
class NumberToken:
    text: str
    value: float
    min_sig: int
    max_sig: int
    is_percent: bool


def _sig_range(literal: str) -> tuple[int, int]:
    """Plausible significant-figure range for a numeric literal (commas stripped).

    Decimals are unambiguous ("1.20" is 3). Integer trailing zeros are
    ambiguous ("100" could be 1-3), so a range is returned and any count in
    it may justify a match — permissive, but deterministic.
    """
    if "." in literal:
        digits = literal.replace(".", "").lstrip("0")
        count = max(len(digits), 1)
        return count, count
    digits = literal.lstrip("0") or "0"
    return max(len(digits.rstrip("0")), 1), max(len(digits), 1)


def extract_numbers(text: str) -> list[NumberToken]:
    """Every number stated in `text`, as displayed-value tokens."""
    cleaned = _ENUMERATOR_RE.sub("", text)
    tokens: list[NumberToken] = []
    for match in _TOKEN_RE.finditer(cleaned):
        end = match.end()
        if end < len(cleaned) and cleaned[end] == "(":  # 401(k), 403(b): names, not numbers
            continue
        literal = match.group("digits").replace(",", "")
        scale = _SCALES[match.group("scale").lower()] if match.group("scale") else 1.0
        min_sig, max_sig = _sig_range(literal)
        tokens.append(
            NumberToken(
                text=match.group(0).strip(),
                value=float(literal) * scale,
                min_sig=min_sig,
                max_sig=max_sig,
                is_percent=match.group("percent") is not None,
            )
        )
    return tokens


def _round_sig(value: float, sig: int) -> float:
    if value == 0:
        return 0.0
    return round(value, sig - 1 - math.floor(math.log10(abs(value))))


def matches_source(token: NumberToken, source: float) -> bool:
    """True if `token` equals `source` rounded to the displayed significant digits.

    A percent token is also compared against source*100, so "15%" matches a
    source stored as 0.15.
    """
    candidates = (source, source * 100) if token.is_percent else (source,)
    for candidate in candidates:
        for sig in range(token.min_sig, token.max_sig + 1):
            if math.isclose(_round_sig(candidate, sig), token.value, rel_tol=1e-9, abs_tol=1e-9):
                return True
    return False


def matches_any(token: NumberToken, sources: tuple[float, ...]) -> bool:
    """True if `token` matches at least one source value."""
    return any(matches_source(token, source) for source in sources)


def numbers_in(obj: object) -> Iterator[float]:
    """Yield every number reachable in `obj`: ints/floats in nested dicts and
    lists, plus numbers stated inside strings (e.g. an infeasibility reason)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        yield from (token.value for token in extract_numbers(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from numbers_in(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from numbers_in(value)
