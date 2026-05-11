"""
Routing functions that determine which branches run after a given node.
This is the only place that needs to change when branching conditions evolve —
node logic and graph topology stay untouched.
"""
from lang_graph_state.domain.state import GraphState


def route_after_mc(state: GraphState) -> list[str]:
    # standard_explanation always runs — it is the baseline overview regardless of MC outcome.
    targets = ["run_standard_explanation"]

    # Contribution and withdrawal are conditional on mc_result.
    # Stub: both are always included until MCResult has real fields to inspect.
    # Example conditional logic:
    # confidence = state.mc_result.confidence_score
    # if confidence < 0.80:
    #     targets.append("run_contribution_explanation")
    # if confidence > 0.95:
    #     targets.append("run_withdrawal_explanation")
    targets.append("run_contribution_explanation")
    targets.append("run_withdrawal_explanation")

    return targets
