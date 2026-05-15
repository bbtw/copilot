import httpx

from lang_graph_state.errors import SourceFetchError
from lang_graph_state.state import GraphState, SynthesisOutput
from lang_graph_state.token_manager import TokenManager


def make_synthesis_node(client: httpx.AsyncClient, token_manager: TokenManager, model_provider: str, model_id: str):
    async def node(state: GraphState) -> dict:
        try:
            token = await token_manager.get_token()
            # TODO: build prompt from state fields and parse response into SynthesisOutput
            _ = token
            return {"synthesis": SynthesisOutput()}
        except Exception as e:
            raise SourceFetchError("synthesis", e) from e
    return node
