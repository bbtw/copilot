"""Federal tax data and model defaults.

All figures are 2026 values (Rev. Proc. 2025-32 brackets/standard deduction,
IRS retirement plan limits, Pub. 590-B Uniform Lifetime Table) and are held
constant across all model years per ADR-0001: the model works in real
(today's) dollars and the IRS indexes these figures to inflation.
"""

from enum import Enum


class FilingStatus(Enum):
    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"


# (upper bound of taxable income, marginal rate); last upper bound is infinite
_BRACKETS: dict[FilingStatus, list[tuple[float, float]]] = {
    FilingStatus.SINGLE: [
        (12_400, 0.10),
        (50_400, 0.12),
        (105_700, 0.22),
        (201_775, 0.24),
        (256_225, 0.32),
        (640_600, 0.35),
        (float("inf"), 0.37),
    ],
    FilingStatus.MARRIED_FILING_JOINTLY: [
        (24_800, 0.10),
        (100_800, 0.12),
        (211_400, 0.22),
        (403_550, 0.24),
        (512_450, 0.32),
        (768_700, 0.35),
        (float("inf"), 0.37),
    ],
}

_STANDARD_DEDUCTION: dict[FilingStatus, float] = {
    FilingStatus.SINGLE: 16_100.0,
    FilingStatus.MARRIED_FILING_JOINTLY: 32_200.0,
}

RMD_START_AGE = 73

# IRS Uniform Lifetime Table (Pub. 590-B); ages past the table clamp to the last entry
_RMD_DIVISORS: dict[int, float] = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}

# 401(k) employee deferral limit; the single combined Traditional+Roth cap (see CONTEXT.md)
DEFAULT_CONTRIBUTION_LIMIT = 24_500.0
DEFAULT_CAPITAL_GAINS_RATE = 0.15
DEFAULT_REAL_RETURN = 0.05
SS_TAXABLE_FRACTION = 0.85
# Annual-granularity stand-in for the age-59½ rule
EARLY_WITHDRAWAL_AGE = 60


def rmd_divisor(age: int) -> float:
    """Uniform Lifetime Table divisor for `age` (>= RMD_START_AGE)."""
    return _RMD_DIVISORS[min(age, max(_RMD_DIVISORS))]


def ordinary_segments(filing_status: FilingStatus) -> list[tuple[float, float]]:
    """Convex piecewise-linear tax segments over gross ordinary income.

    Returns (width, rate) pairs: the standard deduction as a leading 0%-rate
    segment, then each federal bracket. Marginal rates only increase, which is
    what keeps the optimization a pure LP (ADR-0001).
    """
    segments = [(_STANDARD_DEDUCTION[filing_status], 0.0)]
    lower = 0.0
    for upper, rate in _BRACKETS[filing_status]:
        segments.append((upper - lower, rate))
        lower = upper
    return segments


def tax_on_ordinary_income(income: float, filing_status: FilingStatus) -> float:
    """Federal tax on gross ordinary income (standard deduction applied)."""
    remaining = income
    tax = 0.0
    for width, rate in ordinary_segments(filing_status):
        taken = min(remaining, width)
        tax += taken * rate
        remaining -= taken
        if remaining <= 0:
            break
    return tax
