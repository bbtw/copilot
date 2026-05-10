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
    analysis_sections: Annotated[list[AnalysisSection], merge_analysis_sections] = Field(
        default_factory=list,
    )
    explanation: str = ""
    final_output: FinalOutput | None = None
