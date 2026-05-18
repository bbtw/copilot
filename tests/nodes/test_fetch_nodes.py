import httpx
import pytest
from pydantic import BaseModel

from lang_graph_state.errors import SourceFetchError
from lang_graph_state.nodes.fetch_nodes import make_fetch_node
from lang_graph_state.request_spec import RequestSpec
from lang_graph_state.state import GraphState


class FakeResult(BaseModel):
    value: str


class FakeOAuthIdentity:
    def __init__(self, token: str = "test-token") -> None:
        self.token = token

    async def get_token(self) -> str:
        return self.token


def _build_request(state: GraphState) -> RequestSpec:
    return RequestSpec(
        method="GET",
        path="/foo",
        params={"q": "1"},
        headers={"X-Extra": "hi"},
    )


def _parse_response(response: httpx.Response) -> FakeResult:
    return FakeResult(value=response.json()["v"])


def _state() -> GraphState:
    return GraphState(fs_req_id="req-123")


@pytest.mark.asyncio
async def test_fetch_node_happy_path_returns_parsed_under_source_name() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"v": "hello"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    node = make_fetch_node("test_source", _build_request, _parse_response, client, FakeOAuthIdentity("tok-abc"))

    result = await node(_state())

    assert result == {"test_source": FakeResult(value="hello")}

    assert len(captured) == 1
    req = captured[0]
    assert req.method == "GET"
    assert req.url.path == "/foo"
    assert req.url.params["q"] == "1"
    assert req.headers["Authorization"] == "Bearer tok-abc"
    assert req.headers["fsreqid"] == "req-123"
    assert req.headers["X-Extra"] == "hi"

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_node_wraps_http_error_in_source_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    node = make_fetch_node("customer_profile", _build_request, _parse_response, client, FakeOAuthIdentity())

    with pytest.raises(SourceFetchError) as exc:
        await node(_state())

    assert exc.value.source == "customer_profile"
    assert isinstance(exc.value.cause, httpx.HTTPStatusError)

    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_node_wraps_parse_error_in_source_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    def parse_response(response: httpx.Response) -> FakeResult:
        return FakeResult(value=response.json()["v"])  # KeyError

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    node = make_fetch_node("insights", _build_request, parse_response, client, FakeOAuthIdentity())

    with pytest.raises(SourceFetchError) as exc:
        await node(_state())

    assert exc.value.source == "insights"
    assert isinstance(exc.value.cause, KeyError)

    await client.aclose()
