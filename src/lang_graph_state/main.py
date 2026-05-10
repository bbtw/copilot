from dotenv import load_dotenv
load_dotenv()

import asyncio
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from lang_graph_state.domain.state import GraphState
from lang_graph_state.instrumentation.logging import configure_logging
from lang_graph_state.instrumentation.tracing import build_invoke_config
from lang_graph_state.nodes.format_output import format_output
from lang_graph_state.nodes.load_profile import load_profile
from lang_graph_state.nodes.lp_optimizer import run_lp_optimizer
from lang_graph_state.nodes.monte_carlo import run_monte_carlo
from lang_graph_state.nodes.sections import build_section_a_node, build_section_b_node, build_section_c_node
from lang_graph_state.nodes.synthesize_explanation import build_synthesize_explanation_node
from lang_graph_state.services.explanation import ExplanationService


def build_graph(
    explanation_service: ExplanationService,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("load_profile", load_profile)
    graph.add_node("run_lp_optimizer", run_lp_optimizer)
    graph.add_node("run_monte_carlo", run_monte_carlo)

    graph.add_node("run_section_a", build_section_a_node(explanation_service))
    graph.add_node("run_section_b", build_section_b_node(explanation_service))
    graph.add_node("run_section_c", build_section_c_node(explanation_service))

    graph.add_node("synthesize_explanation", build_synthesize_explanation_node(explanation_service))
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "load_profile")
    graph.add_edge("load_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")

    graph.add_edge("run_monte_carlo", "run_section_a")
    graph.add_edge("run_monte_carlo", "run_section_b")
    graph.add_edge("run_monte_carlo", "run_section_c")

    graph.add_edge("run_section_a", "synthesize_explanation")
    graph.add_edge("run_section_b", "synthesize_explanation")
    graph.add_edge("run_section_c", "synthesize_explanation")

    graph.add_edge("synthesize_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile(checkpointer=checkpointer)


async def amain() -> None:
    import uuid
    from langgraph.checkpoint.memory import MemorySaver

    configure_logging()
    app = build_graph(ExplanationService(), checkpointer=MemorySaver())
    await app.ainvoke({}, config=build_invoke_config(thread_id=str(uuid.uuid4())))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
