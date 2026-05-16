from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel

from lang_graph_state.request_spec import RequestSpec, SourceDefinition

if TYPE_CHECKING:
    from lang_graph_state.state import GraphState


class MonteCarloResult(BaseModel):
    pass


def build_request(state: GraphState) -> RequestSpec:
    return RequestSpec(method="GET", path="/")


def parse_response(response: httpx.Response) -> MonteCarloResult:
    return MonteCarloResult()


definition = SourceDefinition(
    name="monte_carlo",
    result_model=MonteCarloResult,
    build_request=build_request,
    parse_response=parse_response,
)
