from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from state import RetirementPlanState
from nodes.load_profile import load_customer_profile
from nodes.lp_optimizer import run_lp_optimizer
from nodes.monte_carlo import run_monte_carlo
from nodes.explanation import build_explanation_node
from nodes.format_output import format_output
from services.explanation import ExplanationService


def route_by_confidence_band(state: RetirementPlanState) -> str:
    return state.confidence_band


def build_graph(
    explanation_service: ExplanationService,
) -> StateGraph:
    graph = StateGraph(RetirementPlanState)

    graph.add_node("load_customer_profile", load_customer_profile)
    graph.add_node("run_lp_optimizer", run_lp_optimizer)
    graph.add_node("run_monte_carlo", run_monte_carlo)
    graph.add_node(
        "generate_low_confidence_explanation",
        build_explanation_node("low", explanation_service),
    )
    graph.add_node(
        "generate_medium_confidence_explanation",
        build_explanation_node("medium", explanation_service),
    )
    graph.add_node(
        "generate_high_confidence_explanation",
        build_explanation_node("high", explanation_service),
    )
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "load_customer_profile")
    graph.add_edge("load_customer_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")
    graph.add_conditional_edges(
        "run_monte_carlo",
        route_by_confidence_band,
        {
            "low": "generate_low_confidence_explanation",
            "medium": "generate_medium_confidence_explanation",
            "high": "generate_high_confidence_explanation",
        },
    )
    graph.add_edge("generate_low_confidence_explanation", "format_output")
    graph.add_edge("generate_medium_confidence_explanation", "format_output")
    graph.add_edge("generate_high_confidence_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


if __name__ == "__main__":
    _explanation_service = ExplanationService()
    app = build_graph(_explanation_service)
    final_state = app.invoke({})
    final_state["result"].print_summary()
