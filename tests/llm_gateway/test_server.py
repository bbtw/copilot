import json

import httpx
from fastapi.testclient import TestClient

from lang_graph_state.llm_gateway.server import GatewaySettings, create_app


def _settings() -> GatewaySettings:
    return GatewaySettings(
        upstream_base_url="http://upstream.test/v1",
        upstream_api_key="test-secret",
        host="127.0.0.1",
        port=8001,
    )


def test_healthz_omits_api_key():
    client = TestClient(create_app(_settings()))

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "gateway": {
            "host": "127.0.0.1",
            "port": 8001,
        },
        "upstream": {
            "base_url": "http://upstream.test/v1",
            "auth_configured": True,
        },
    }
    assert "test-secret" not in response.text


def test_chat_completions_forwards_request_and_returns_upstream_json():
    upstream_requests: list[dict] = []
    upstream_response = {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_requests.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "json": json.loads(request.content.decode()),
        })
        return httpx.Response(200, json=upstream_response)

    client = TestClient(create_app(_settings(), transport=httpx.MockTransport(handler)))
    request_body = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
    }

    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    assert response.json() == upstream_response
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["url"] == "http://upstream.test/v1/chat/completions"
    assert upstream_requests[0]["headers"]["authorization"] == "Bearer test-secret"
    assert upstream_requests[0]["json"] == request_body


def test_chat_completions_rejects_streaming_requests():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("streaming requests should not reach upstream")

    client = TestClient(create_app(_settings(), transport=httpx.MockTransport(handler)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "stream=true is not supported by the local LLM gateway"


def test_chat_completions_maps_upstream_500():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "ollama failed"}})

    client = TestClient(create_app(_settings(), transport=httpx.MockTransport(handler)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "ollama failed"


def test_chat_completions_maps_connection_failure_to_502():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed to connect", request=request)

    client = TestClient(create_app(_settings(), transport=httpx.MockTransport(handler)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Upstream LLM gateway is unavailable"
