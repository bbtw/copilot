"""
Placeholder domain models. Field shapes are TBD when production APIs are wired.
The reducer is defined here because it is tightly coupled to AnalysisSection and its ordering rules.
"""
from typing import Literal

from pydantic import BaseModel

AnalysisKind = Literal["section_a", "section_b", "section_c"]
ANALYSIS_SECTION_ORDER: dict[str, int] = {
    "section_a": 0,
    "section_b": 1,
    "section_c": 2,
}


class Profile(BaseModel):
    pass


class LPResult(BaseModel):
    pass


class MCResult(BaseModel):
    pass


class AnalysisSection(BaseModel):
    kind: AnalysisKind
    content: str


class FinalOutput(BaseModel):
    explanation: str = ""


def merge_analysis_sections(
    current: list[AnalysisSection] | None,
    updates: list[AnalysisSection] | None,
) -> list[AnalysisSection]:
    # Deduplication by kind makes retries idempotent: a replayed branch overwrites rather than appends.
    by_kind = {section.kind: section for section in current or []}
    for section in updates or []:
        by_kind[section.kind] = section
    return sorted(by_kind.values(), key=lambda s: ANALYSIS_SECTION_ORDER[s.kind])
