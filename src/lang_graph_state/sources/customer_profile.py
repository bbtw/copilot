from __future__ import annotations

import httpx
from pydantic import BaseModel

from lang_graph_state.request_spec import RequestSpec, RequestState, SourceDefinition


class CustomerProfileResult(BaseModel):
    pass


def build_request(state: RequestState) -> RequestSpec:
    return RequestSpec(method="GET", path="/")


def parse_response(response: httpx.Response) -> CustomerProfileResult:
    return CustomerProfileResult()


definition = SourceDefinition(
    name="customer_profile",
    result_model=CustomerProfileResult,
    build_request=build_request,
    parse_response=parse_response,
)
