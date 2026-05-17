
from typing import Awaitable, Callable

import httpx

from lang_graph_state.errors import SourceFetchError
from lang_graph_state.request_spec import SourceDefinition
from lang_graph_state.state import GraphState
from lang_graph_state.token_manager import TokenManager

Node = Callable[[GraphState], Awaitable[dict]]


def make_fetch_node(
    source: SourceDefinition,
    client: httpx.AsyncClient,
    token_manager: TokenManager,
) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            spec = source.build_request(state)
            token = await token_manager.get_token()
            headers = dict(spec.headers or {})
            headers["Authorization"] = f"Bearer {token}"
            headers["fsreqid"] = state.fs_req_id
            response = await client.request(
                method=spec.method,
                url=spec.path,
                json=spec.json,
                params=spec.params,
                headers=headers,
            )
            response.raise_for_status()
            parsed = source.parse_response(response)
            return {source.name: parsed}
        except Exception as e:
            raise SourceFetchError(source.name, e) from e

    return node
