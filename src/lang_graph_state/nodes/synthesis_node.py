from __future__ import annotations

from typing import Awaitable, Callable

import httpx

from lang_graph_state.errors import SynthesisError
from lang_graph_state.request_spec import RequestSpec
from lang_graph_state.state import GraphState
from lang_graph_state.synthesis.parsing import SynthesisOutput
from lang_graph_state.token_manager import TokenManager

Node = Callable[[GraphState], Awaitable[dict]]
BuildRequest = Callable[[GraphState, str, str], RequestSpec]
ParseResponse = Callable[[httpx.Response], SynthesisOutput]


def make_synthesis_node(
    client: httpx.AsyncClient,
    token_manager: TokenManager,
    model_provider: str,
    model_id: str,
    build_request: BuildRequest,
    parse_response: ParseResponse,
) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            spec = build_request(state, model_provider, model_id)
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
            parsed = parse_response(response)
            return {"synthesis": parsed}
        except Exception as e:
            raise SynthesisError(e) from e

    return node
