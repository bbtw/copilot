from langgraph.graph import StateGraph, START, END
from state import RetirementPlanState
from nodes.load_profile import load_customer_profile
from nodes.lp_optimizer import run_lp_optimizer
from nodes.monte_carlo import run_monte_carlo
from nodes.explanation import generate_explanation
from nodes.format_output import format_output


def build_graph() -> StateGraph:
    graph = StateGraph(RetirementPlanState)

    graph.add_node("load_customer_profile", load_customer_profile)
    graph.add_node("run_lp_optimizer", run_lp_optimizer)
    graph.add_node("run_monte_carlo", run_monte_carlo)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "load_customer_profile")
    graph.add_edge("load_customer_profile", "run_lp_optimizer")
    graph.add_edge("run_lp_optimizer", "run_monte_carlo")
    graph.add_edge("run_monte_carlo", "generate_explanation")
    graph.add_edge("generate_explanation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    final_state = app.invoke({})
    final_state["result"].print_summary()
