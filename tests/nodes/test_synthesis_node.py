import httpx
import pytest
from pydantic import BaseModel

from lang_graph_state.errors import SynthesisError
from lang_graph_state.nodes.synthesis_node import make_synthesis_node
from lang_graph_state.request_spec import RequestSpec
from lang_graph_state.state import GraphState


class FakeOutput(BaseModel):
    answer: str


class FakeOAuthIdentity:
    def __init__(self, token: str = "test-token") -> None:
        self.token = token

    async def get_token(self) -> str:
        return self.token


def _state() -> GraphState:
    return GraphState(fs_req_id="req-xyz")


@pytest.mark.asyncio
async def test_synthesis_node_happy_path_returns_parsed_under_synthesis() -> None:
    captured: list[httpx.Request] = []
    captured_args: dict[str, object] = {}

    def build_request(state: GraphState, provider: str, model_id: str) -> RequestSpec:
        captured_args["state"] = state
        captured_args["provider"] = provider
        captured_args["model_id"] = model_id
        return RequestSpec(
            method="POST",
            path="/v1/complete",
            json={
                "model": {"provider": provider, "id": model_id},
                "prompt_spec": {"messages": [{"role": "user", "content": "hi"}]},
            },
            headers={"X-Trace": "abc"},
        )

    def parse_response(response: httpx.Response) -> FakeOutput:
        return FakeOutput(answer=response.json()["answer"])

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"answer": "42"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    node = make_synthesis_node(
        client,
        FakeOAuthIdentity("tok-syn"),
        "anthropic",
        "claude-opus-4-7",
        build_request,
        parse_response,
    )

    state = _state()
    result = await node(state)

    assert result == {"synthesis": FakeOutput(answer="42")}

    assert captured_args == {
        "state": state,
        "provider": "anthropic",
        "model_id": "claude-opus-4-7",
    }

    assert len(captured) == 1
    req = captured[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/complete"
    assert req.headers["Authorization"] == "Bearer tok-syn"
    assert req.headers["fsreqid"] == "req-xyz"
    assert req.headers["X-Trace"] == "abc"

    await client.aclose()


@pytest.mark.asyncio
async def test_synthesis_node_wraps_http_error_in_synthesis_error() -> None:
    def build_request(state: GraphState, provider: str, model_id: str) -> RequestSpec:
        return RequestSpec(method="POST", path="/")

    def parse_response(response: httpx.Response) -> FakeOutput:
        return FakeOutput(answer=response.json()["answer"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    node = make_synthesis_node(
        client,
        FakeOAuthIdentity(),
        "anthropic",
        "claude-opus-4-7",
        build_request,
        parse_response,
    )

    with pytest.raises(SynthesisError) as exc:
        await node(_state())

    assert isinstance(exc.value.cause, httpx.HTTPStatusError)

    await client.aclose()


@pytest.mark.asyncio
async def test_synthesis_node_wraps_parse_error_in_synthesis_error() -> None:
    def build_request(state: GraphState, provider: str, model_id: str) -> RequestSpec:
        return RequestSpec(method="POST", path="/")

    def parse_response(response: httpx.Response) -> FakeOutput:
        return FakeOutput(answer=response.json()["answer"])  # KeyError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    node = make_synthesis_node(
        client,
        FakeOAuthIdentity(),
        "anthropic",
        "claude-opus-4-7",
        build_request,
        parse_response,
    )

    with pytest.raises(SynthesisError) as exc:
        await node(_state())

    assert isinstance(exc.value.cause, KeyError)

    await client.aclose()
