from lang_graph_state.request_spec import RequestSpec
from lang_graph_state.state import GraphState


def build_synthesis_request(state: GraphState, model_provider: str, model_id: str) -> RequestSpec:
    return RequestSpec(
        method="POST",
        path="/",
        json={
            "model": {"provider": model_provider, "id": model_id},
            "prompt_spec": {"messages": []},
        },
    )
