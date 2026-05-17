
import httpx
from pydantic import BaseModel

from lang_graph_state.request_spec import RequestSpec, SourceDefinition
from lang_graph_state.state import GraphState


class OptimizerResult(BaseModel):
    pass


def build_request(state: GraphState) -> RequestSpec:
    return RequestSpec(method="GET", path="/")


def parse_response(response: httpx.Response) -> OptimizerResult:
    return OptimizerResult()


definition = SourceDefinition(
    name="optimizer",
    result_model=OptimizerResult,
    build_request=build_request,
    parse_response=parse_response,
)
