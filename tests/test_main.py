import asyncio
from unittest.mock import AsyncMock, MagicMock

from lang_graph_state.main import build_graph
from lang_graph_state.services.gateway import GatewayClient


def _mock_client(response: str = "stub") -> MagicMock:
    mock = MagicMock(spec=GatewayClient)
    mock.acomplete = AsyncMock(return_value=response)
    return mock


def test_build_graph_has_expected_nodes():
    app = build_graph(_mock_client())
    node_names = set(app.get_graph().nodes.keys())
    assert "run_standard_explanation" in node_names
    assert "run_contribution_explanation" in node_names
    assert "run_withdrawal_explanation" in node_names
    assert "synthesize_explanation" in node_names
    assert "format_output" in node_names


def test_graph_runs_end_to_end():
    app = build_graph(_mock_client("analysis"))
    final_state = asyncio.run(app.ainvoke({}))
    assert final_state["final_output"] is not None
    assert {s.kind for s in final_state["analysis_sections"]} == {
        "standard_explanation",
        "contribution_explanation",
        "withdrawal_explanation",
    }
    assert final_state["explanation"] == "analysis"
