from __future__ import annotations

import httpx
from pydantic import BaseModel

from lang_graph_state.request_spec import RequestSpec, RequestState, SourceDefinition


class MonteCarloResult(BaseModel):
    pass


def build_request(state: RequestState) -> RequestSpec:
    return RequestSpec(method="GET", path="/")


def parse_response(response: httpx.Response) -> MonteCarloResult:
    return MonteCarloResult()


definition = SourceDefinition(
    name="monte_carlo",
    result_model=MonteCarloResult,
    build_request=build_request,
    parse_response=parse_response,
)
