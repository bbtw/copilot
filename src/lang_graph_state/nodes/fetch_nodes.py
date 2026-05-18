
from typing import Awaitable, Callable

import httpx

from lang_graph_state.errors import SourceFetchError
from lang_graph_state.oauth_identity import OAuthIdentity
from lang_graph_state.request_spec import RequestSpec, RequestState
from lang_graph_state.state import GraphState

Node = Callable[[GraphState], Awaitable[dict]]


def make_fetch_node(
    name: str,
    build_request: Callable[[RequestState], RequestSpec],
    parse_response: Callable[[httpx.Response], object],
    client: httpx.AsyncClient,
    identity: OAuthIdentity,
) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            spec = build_request(state)
            token = await identity.get_token()
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
            parsed = parse_response(response)
            return {name: parsed}
        except Exception as e:
            raise SourceFetchError(name, e) from e

    return node
