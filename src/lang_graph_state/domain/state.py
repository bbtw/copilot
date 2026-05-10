"""
LangGraph graph state. Nodes receive the full state and return only the keys they own.
analysis_sections is the only field with a reducer because three nodes write to it concurrently during the fan-out.
"""
from typing import Annotated

from pydantic import BaseModel, Field

from lang_graph_state.domain.models import (
    AnalysisSection,
    FinalOutput,
    LPResult,
    MCResult,
    Profile,
    merge_analysis_sections,
)


class GraphState(BaseModel):
    profile: Profile | None = None
    lp_result: LPResult | None = None
    mc_result: MCResult | None = None
    # Reducer is required here: without it LangGraph raises a conflict error when the three fan-out branches write concurrently.
    analysis_sections: Annotated[list[AnalysisSection], merge_analysis_sections] = Field(
        default_factory=list,
    )
    explanation: str = ""
    final_output: FinalOutput | None = None
