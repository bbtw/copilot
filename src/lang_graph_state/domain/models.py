"""
Placeholder domain models. Field shapes are TBD when production APIs are wired.
The reducer is defined here because it is tightly coupled to AnalysisSection and its ordering rules.
"""
from typing import Literal

from pydantic import BaseModel

# The three analysis branches that can run in the fan-out. standard_explanation always runs;
# contribution_explanation and withdrawal_explanation are conditional on mc_result via route_after_mc.
# Adding a new branch means: add a kind here, add its order below, add a builder in nodes/sections.py,
# and wire it in nodes/routing.py and main.py.
AnalysisKind = Literal["standard_explanation", "contribution_explanation", "withdrawal_explanation"]

# Controls the order of sections in the final output regardless of which branches completed first.
# synthesize_explanation receives sections in this order so the synthesized explanation is consistent
# across invocations even when branches finish at different times.
ANALYSIS_SECTION_ORDER: dict[str, int] = {
    "standard_explanation": 0,
    "contribution_explanation": 1,
    "withdrawal_explanation": 2,
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
    # LangGraph calls this reducer each time any fan-out branch writes to analysis_sections.
    # Keying by kind serves two purposes: concurrent writes from different branches are merged
    # cleanly, and retried branches overwrite rather than append — keeping state idempotent.
    by_kind = {section.kind: section for section in current or []}
    for section in updates or []:
        by_kind[section.kind] = section
    # Sort by ANALYSIS_SECTION_ORDER so the list is stable regardless of branch completion order.
    return sorted(by_kind.values(), key=lambda s: ANALYSIS_SECTION_ORDER[s.kind])
