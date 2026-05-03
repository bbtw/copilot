from collections.abc import Callable
from typing import Any, cast

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from state import RetirementPlanState
from nodes.load_profile import load_customer_profile
from nodes.lp_optimizer import run_lp_optimizer
from nodes.monte_carlo import build_monte_carlo_node
from nodes.explanation import build_explanation_node
from nodes.format_output import format_output
from services.explanation import ExplanationService

StatePatch = dict[str, Any]
GraphNode = Callable[[RetirementPlanState], StatePatch]


def as_langgraph_node(node: GraphNode) -> Any:
    return cast(Any, node)


def route_by_confidence_band(state: RetirementPlanState) -> str:
    if state.confidence_band is None:
        raise ValueError("Confidence band must be set before routing.")
    return state.confidence_band


def build_graph(
    explanation_service: ExplanationService,
    *,
    rng: np.random.Generator | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    # Initialize the state graph with our custom state schema
    graph = StateGraph(RetirementPlanState)

    graph.add_node("load_customer_profile", as_langgraph_node(load_customer_profile))
    graph.add_node("run_lp_optimizer", as_langgraph_node(run_lp_optimizer))
    graph.add_node("run_monte_carlo", as_langgraph_node(build_monte_carlo_node(rng)))
    graph.add_node(
        "generate_low_confidence_explanation",
        as_langgraph_node(build_explanation_node("low", explanation_service)),
    )
    graph.add_node(
        "generate_medium_confidence_explanation",
        as_langgraph_node(build_explanation_node("medium", explanation_service)),
    )
    graph.add_node(
        "generate_high_confidence_explanation",
        as_langgraph_node(build_explanation_node("high", explanation_service)),
    )
    graph.add_node("format_output", as_langgraph_node(format_output))

    # Define the core linear flow: Load profile -> Run Optimizer -> Run Monte Carlo
    graph.add_edge(START, "load_customer_profile")
    graph.add_edge("load_customer_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")

    # Conditional Routing: After running the Monte Carlo simulation, we branch the graph.
    # The 'route_by_confidence_band' function inspects the state's confidence band.
    # Depending on whether the band is 'low', 'medium', or 'high', the flow is directed
    # to a specific explanation node tailored to that confidence level.
    graph.add_conditional_edges(
        "run_monte_carlo",
        route_by_confidence_band,
        {
            "low": "generate_low_confidence_explanation",
            "medium": "generate_medium_confidence_explanation",
            "high": "generate_high_confidence_explanation",
        },
    )

    # Reconvergence: All explanation paths merge back into 'format_output'
    # to prepare the final response before reaching the END.
    graph.add_edge("generate_low_confidence_explanation", "format_output")
    graph.add_edge("generate_medium_confidence_explanation", "format_output")
    graph.add_edge("generate_high_confidence_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import uuid
    from langgraph.checkpoint.memory import MemorySaver
    _explanation_service = ExplanationService()
    app = build_graph(_explanation_service, checkpointer=MemorySaver())
    final_state = app.invoke({}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    final_state["result"].print_summary()
