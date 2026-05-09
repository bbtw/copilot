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
from nodes.standard_analysis import build_standard_analysis_node
from nodes.accumulation_agent import build_accumulation_node
from nodes.withdrawal_agent import build_withdrawal_node
from nodes.synthesize_explanation import build_synthesize_explanation_node
from nodes.format_output import format_output
from services.explanation import ExplanationService

StatePatch = dict[str, Any]
GraphNode = Callable[[RetirementPlanState], StatePatch]


def as_node(node: GraphNode) -> Any:
    return cast(Any, node)


def build_graph(
    explanation_service: ExplanationService,
    *,
    rng: np.random.Generator | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(RetirementPlanState)

    # Data gathering — sequential
    graph.add_node("load_customer_profile", as_node(load_customer_profile))
    graph.add_node("run_lp_optimizer", as_node(run_lp_optimizer))
    graph.add_node("run_monte_carlo", as_node(build_monte_carlo_node(rng)))

    # Parallel analysis — fan-out from run_monte_carlo
    graph.add_node("run_standard_analysis", as_node(build_standard_analysis_node(explanation_service)))
    graph.add_node("run_accumulation_agent", as_node(build_accumulation_node()))
    graph.add_node("run_withdrawal_agent", as_node(build_withdrawal_node()))

    # Synthesis and output — fan-in
    graph.add_node("synthesize_explanation", as_node(build_synthesize_explanation_node(explanation_service)))
    graph.add_node("format_output", as_node(format_output))

    # Sequential data gathering flow
    graph.add_edge(START, "load_customer_profile")
    graph.add_edge("load_customer_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")

    # Fan-out: all three agents start after Monte Carlo completes
    graph.add_edge("run_monte_carlo", "run_standard_analysis")
    graph.add_edge("run_monte_carlo", "run_accumulation_agent")
    graph.add_edge("run_monte_carlo", "run_withdrawal_agent")

    # Fan-in: synthesis waits for all three to write their state fields
    graph.add_edge("run_standard_analysis", "synthesize_explanation")
    graph.add_edge("run_accumulation_agent", "synthesize_explanation")
    graph.add_edge("run_withdrawal_agent", "synthesize_explanation")

    graph.add_edge("synthesize_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import uuid
    from langgraph.checkpoint.memory import MemorySaver

    _explanation_service = ExplanationService()
    app = build_graph(_explanation_service, checkpointer=MemorySaver())
    final_state = app.invoke({}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    final_state["result"].print_summary()
