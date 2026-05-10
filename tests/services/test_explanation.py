import asyncio
from unittest.mock import AsyncMock, MagicMock

from lang_graph_state.domain.models import AnalysisPayload, AnalysisSection
from lang_graph_state.services.explanation import ExplanationService
from lang_graph_state.services.llm import GatewayClient


def _mock_gateway(response: str = "stub") -> MagicMock:
    mock = MagicMock(spec=GatewayClient)
    mock.acomplete = AsyncMock(return_value=response)
    return mock


def test_aanalyze_returns_gateway_response():
    gateway = _mock_gateway("analysis result")
    service = ExplanationService(llm=gateway)
    result = asyncio.run(service.aanalyze("section_a", AnalysisPayload()))
    assert result == "analysis result"
    gateway.acomplete.assert_called_once()


def test_aanalyze_calls_gateway_per_kind():
    gateway = _mock_gateway()
    service = ExplanationService(llm=gateway)
    asyncio.run(service.aanalyze("section_b", AnalysisPayload()))
    asyncio.run(service.aanalyze("section_c", AnalysisPayload()))
    assert gateway.acomplete.call_count == 2


def test_asynthesize_returns_gateway_response():
    gateway = _mock_gateway("final explanation")
    service = ExplanationService(llm=gateway)
    sections = [
        AnalysisSection(kind="section_a", content="a"),
        AnalysisSection(kind="section_b", content="b"),
        AnalysisSection(kind="section_c", content="c"),
    ]
    result = asyncio.run(service.asynthesize(sections))
    assert result == "final explanation"
    gateway.acomplete.assert_called_once()
