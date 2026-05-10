from dotenv import load_dotenv
load_dotenv()

import asyncio
import numpy as np
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from lang_graph_state.domain.state import RetirementPlanState
from lang_graph_state.instrumentation.logging import configure_logging
from lang_graph_state.instrumentation.nodes import instrument_node
from lang_graph_state.instrumentation.tracing import build_invoke_config
from lang_graph_state.nodes.load_profile import load_customer_profile
from lang_graph_state.nodes.lp_optimizer import run_lp_optimizer
from lang_graph_state.nodes.monte_carlo import build_monte_carlo_node
from lang_graph_state.nodes.standard_analysis import build_standard_analysis_node
from lang_graph_state.nodes.analysis_agents import build_accumulation_node, build_withdrawal_node
from lang_graph_state.nodes.synthesize_explanation import build_synthesize_explanation_node
from lang_graph_state.nodes.format_output import format_output
from lang_graph_state.services.explanation import ExplanationService


def build_graph(
    explanation_service: ExplanationService,
    *,
    rng: np.random.Generator | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(RetirementPlanState)

    # Sequential data preparation. These placeholder processors are deliberately
    # plain nodes because production will replace them with black-box APIs.
    graph.add_node(
        "load_customer_profile",
        instrument_node("load_customer_profile", load_customer_profile),
    )
    graph.add_node(
        "run_lp_optimizer",
        instrument_node("run_lp_optimizer", run_lp_optimizer),
    )
    graph.add_node(
        "run_monte_carlo",
        instrument_node("run_monte_carlo", build_monte_carlo_node(rng)),
    )

    # Parallel analysis fan-out. Each branch writes one PlanAnalysisSection to
    # the shared reducer-backed `analysis_sections` state field.
    graph.add_node(
        "run_standard_analysis",
        instrument_node(
            "run_standard_analysis",
            build_standard_analysis_node(explanation_service),
        ),
    )
    graph.add_node(
        "run_accumulation_agent",
        instrument_node("run_accumulation_agent", build_accumulation_node(explanation_service)),
    )
    graph.add_node(
        "run_withdrawal_agent",
        instrument_node("run_withdrawal_agent", build_withdrawal_node(explanation_service)),
    )

    # Fan-in and final formatting. LangGraph schedules synthesis only after all
    # incoming parallel writes for the super-step have been merged.
    graph.add_node(
        "synthesize_explanation",
        instrument_node(
            "synthesize_explanation",
            build_synthesize_explanation_node(explanation_service),
        ),
    )
    graph.add_node(
        "format_output",
        instrument_node("format_output", format_output),
    )

    graph.add_edge(START, "load_customer_profile")
    graph.add_edge("load_customer_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")

    graph.add_edge("run_monte_carlo", "run_standard_analysis")
    graph.add_edge("run_monte_carlo", "run_accumulation_agent")
    graph.add_edge("run_monte_carlo", "run_withdrawal_agent")

    graph.add_edge("run_standard_analysis", "synthesize_explanation")
    graph.add_edge("run_accumulation_agent", "synthesize_explanation")
    graph.add_edge("run_withdrawal_agent", "synthesize_explanation")

    graph.add_edge("synthesize_explanation", "format_output")
    graph.add_edge("format_output", END)

    # `checkpointer` persists per-thread graph state; `store` is reserved for
    # cross-thread memory when future interactive workflows need it.
    return graph.compile(checkpointer=checkpointer, store=store)


async def amain() -> None:
    import uuid
    from langgraph.checkpoint.memory import MemorySaver

    configure_logging()
    app = build_graph(ExplanationService(), checkpointer=MemorySaver())
    final_state = await app.ainvoke({}, config=build_invoke_config(thread_id=str(uuid.uuid4())))
    final_state["result"].print_summary()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
