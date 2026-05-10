import asyncio

import numpy as np
from lang_graph_state.main import build_graph
from lang_graph_state.domain.models import PlanAnalysisSection, merge_analysis_sections


class FakeExplanationService:
    async def astandard_analysis(self, request):
        return f"standard analysis for {request.confidence_band}"

    async def aaccumulation_analysis(self, request, probe_results):
        return f"accumulation analysis for {request.confidence_band}: {len(probe_results)} probes"

    async def awithdrawal_analysis(self, request, probe_results):
        return f"withdrawal analysis for {request.confidence_band}: {len(probe_results)} probes"

    async def asynthesize(self, standard, accumulation, withdrawal, confidence_band):
        return f"synthesized for {confidence_band}: {standard[:20]}"


def test_build_graph_compiles_with_correct_nodes():
    app = build_graph(FakeExplanationService(), rng=np.random.default_rng(42))
    node_names = set(app.get_graph().nodes.keys())
    assert "run_standard_analysis" in node_names
    assert "run_accumulation_agent" in node_names
    assert "run_withdrawal_agent" in node_names
    assert "synthesize_explanation" in node_names
    assert "route_by_confidence_band" not in node_names


def test_build_graph_runs_end_to_end_with_injected_service():
    app = build_graph(FakeExplanationService(), rng=np.random.default_rng(42))
    final_state = asyncio.run(app.ainvoke({}))
    result = final_state["result"]
    assert result.explanation.startswith("synthesized for ")
    assert {section.kind for section in final_state["analysis_sections"]} == {
        "standard",
        "accumulation",
        "withdrawal",
    }
    assert result.projected_wealth > 0
    assert result.confidence_score >= 0


def test_analysis_section_reducer_merges_parallel_branch_writes_by_kind():
    current = [PlanAnalysisSection(kind="standard", content="old")]
    updates = [
        PlanAnalysisSection(kind="withdrawal", content="withdrawal"),
        PlanAnalysisSection(kind="standard", content="new"),
    ]

    merged = merge_analysis_sections(current, updates)

    assert [(section.kind, section.content) for section in merged] == [
        ("standard", "new"),
        ("withdrawal", "withdrawal"),
    ]
