"""
Graph assembly and entry point. build_graph wires all nodes and edges; amain runs a single
end-to-end invocation with an in-memory checkpointer.
"""
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
from lang_graph_state.nodes.routing import route_after_mc
from lang_graph_state.nodes.sections import (
    build_standard_explanation_node,
    build_contribution_explanation_node,
    build_withdrawal_explanation_node,
)
from lang_graph_state.nodes.synthesize_explanation import build_synthesize_explanation_node
from lang_graph_state.services.gateway import GatewayClient


def build_graph(
    client: GatewayClient,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("load_profile", load_profile)
    graph.add_node("run_lp_optimizer", run_lp_optimizer)
    graph.add_node("run_monte_carlo", run_monte_carlo)

    graph.add_node("run_standard_explanation", build_standard_explanation_node(client))
    graph.add_node("run_contribution_explanation", build_contribution_explanation_node(client))
    graph.add_node("run_withdrawal_explanation", build_withdrawal_explanation_node(client))

    graph.add_node("synthesize_explanation", build_synthesize_explanation_node(client))
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "load_profile")
    graph.add_edge("load_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")

    # Conditional fan-out: After "run_monte_carlo" finishes, LangGraph pauses and calls route_after_mc.
    # route_after_mc inspects the current state and returns a list of node names to execute next.
    # LangGraph will dynamically route to these nodes and execute them in parallel.
    # The returned node names must exactly match nodes already added to the graph above.
    graph.add_conditional_edges("run_monte_carlo", route_after_mc)

    # These edges define the flow from the explanation nodes to the synthesis node.
    # Note: If an explanation node was NOT triggered by route_after_mc,
    # LangGraph simply ignores its outgoing edge. It will only wait for
    # the branches that DID run to finish before moving to synthesize_explanation.
    graph.add_edge("run_standard_explanation", "synthesize_explanation")
    graph.add_edge("run_contribution_explanation", "synthesize_explanation")
    graph.add_edge("run_withdrawal_explanation", "synthesize_explanation")

    graph.add_edge("synthesize_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile(checkpointer=checkpointer)


async def amain() -> None:
    import uuid
    from langgraph.checkpoint.memory import MemorySaver

    configure_logging()
    app = build_graph(GatewayClient(), checkpointer=MemorySaver())
    await app.ainvoke({}, config=build_invoke_config(thread_id=str(uuid.uuid4())))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
