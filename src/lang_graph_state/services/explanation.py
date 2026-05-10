from lang_graph_state.domain.models import AnalysisKind, AnalysisPayload, AnalysisSection
from lang_graph_state.services.llm import GatewayClient


class ExplanationService:
    def __init__(self, llm: GatewayClient | None = None) -> None:
        self._llm = llm or GatewayClient()

    async def aanalyze(self, kind: AnalysisKind, payload: AnalysisPayload) -> str:
        return await self._llm.acomplete("", system="")

    async def asynthesize(self, sections: list[AnalysisSection]) -> str:
        return await self._llm.acomplete("", system="")
