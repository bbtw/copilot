from copilot.evals.numbers import extract_numbers, matches_any, numbers_in


def _values(text: str) -> list[float]:
    return [token.value for token in extract_numbers(text)]


def _check(text: str, sources: tuple[float, ...]) -> list[bool]:
    return [matches_any(token, sources) for token in extract_numbers(text)]


def test_sigfig_rounding_matches() -> None:
    assert _check("You end with $1.2M after tax.", (1_203_456.0,)) == [True]


def test_wrong_sigfig_fails() -> None:
    assert _check("You end with $1.25M after tax.", (1_203_456.0,)) == [False]


def test_exact_comma_number_matches() -> None:
    assert _check("The limit is $24,500 per year.", (24_500.0,)) == [True]


def test_derived_arithmetic_fails() -> None:
    # $320k total over 8 years: "$40k/year" is derived, so it matches nothing
    assert _check("That's $40k/year.", (320_000.0, 8.0)) == [False]


def test_percent_matches_fraction_source() -> None:
    assert _check("I assumed a 5% real return and 15% capital gains.", (0.05, 0.15)) == [
        True,
        True,
    ]


def test_scale_words() -> None:
    assert _values("about 1.5 million, plus 30 thousand") == [1_500_000.0, 30_000.0]


def test_account_names_not_extracted() -> None:
    assert _values("roll your 401(k) and 403(b) into the traditional bucket") == []


def test_markdown_enumerators_skipped() -> None:
    assert _values("1. Retire at 65\n2. Spend $80k") == [65.0, 80_000.0]


def test_zero_matches_zero() -> None:
    assert _check("You have $0 in Roth today.", (0.0,)) == [True]


def test_trailing_zero_integer_is_permissive() -> None:
    # "$100k" could be 1-3 significant figures; 96,500 rounds to it at 1
    assert _check("roughly $100k", (96_500.0,)) == [True]


def test_numbers_in_walks_nested_values_and_strings() -> None:
    obj = {"a": 65, "b": [{"c": 0.05}], "d": "unlock at age 60", "e": True}
    assert sorted(numbers_in(obj)) == [0.05, 60.0, 65.0]
