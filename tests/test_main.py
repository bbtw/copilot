import numpy as np
from main import build_graph, route_by_confidence_band
from state import RetirementPlanState


def test_route_by_confidence_band_uses_monte_carlo_band():
    assert route_by_confidence_band(RetirementPlanState(confidence_band="low")) == "low"
    assert route_by_confidence_band(RetirementPlanState(confidence_band="medium")) == "medium"
    assert route_by_confidence_band(RetirementPlanState(confidence_band="high")) == "high"


class FakeExplanationService:
    def generate(self, request):
        return f"fake explanation for {request.confidence_band}"


def test_build_graph_accepts_injected_explanation_service():
    app = build_graph(explanation_service=FakeExplanationService(), rng=np.random.default_rng(42))

    final_state = app.invoke({})

    assert final_state["result"].explanation.startswith("fake explanation for ")
