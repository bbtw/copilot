from __future__ import annotations

import httpx
import pytest

from lang_graph_state.errors import SourceFetchError
from lang_graph_state.graph import build_graph, run


SOURCE_HOSTS = {
    "customer-profile.test": "customer_profile",
    "insights.test": "insights",
    "optimizer.test": "optimizer",
    "monte-carlo.test": "monte_carlo",
}


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OAUTH_TOKEN_URL", "http://oauth.test/token")
    monkeypatch.setenv("CUSTOMER_PROFILE_URL", "http://customer-profile.test")
    monkeypatch.setenv("CUSTOMER_PROFILE_USER", "u")
    monkeypatch.setenv("CUSTOMER_PROFILE_PASS", "p")
    monkeypatch.setenv("INSIGHTS_URL", "http://insights.test")
    monkeypatch.setenv("INSIGHTS_USER", "u")
    monkeypatch.setenv("INSIGHTS_PASS", "p")
    monkeypatch.setenv("OPTIMIZER_URL", "http://optimizer.test")
    monkeypatch.setenv("OPTIMIZER_USER", "u")
    monkeypatch.setenv("OPTIMIZER_PASS", "p")
    monkeypatch.setenv("MONTE_CARLO_URL", "http://monte-carlo.test")
    monkeypatch.setenv("MONTE_CARLO_USER", "u")
    monkeypatch.setenv("MONTE_CARLO_PASS", "p")
    monkeypatch.setenv("LLM_GATEWAY_BASE_URL", "http://llm.test")
    monkeypatch.setenv("LLM_GATEWAY_USER", "u")
    monkeypatch.setenv("LLM_GATEWAY_PASS", "p")
    monkeypatch.setenv("LLM_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL_ID", "claude-opus-4-7")
    monkeypatch.setenv("CHECKPOINTER_BACKEND", "memory")


def _as_dict(result: object) -> dict:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return dict(result)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_build_graph_runs_all_sources_then_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        host = request.url.host
        if host == "oauth.test":
            return httpx.Response(200, json={"access_token": "tok"})
        if host in SOURCE_HOSTS or host == "llm.test":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    async with build_graph(transport=transport) as graph:
        result = await run(graph, fs_req_id="req-xyz")

    result_dict = _as_dict(result)
    assert result_dict["fs_req_id"] == "req-xyz"
    for field in ("customer_profile", "insights", "optimizer", "monte_carlo", "synthesis"):
        assert result_dict[field] is not None, f"{field} should be populated"

    by_host: dict[str, list[httpx.Request]] = {}
    for req in requests:
        by_host.setdefault(req.url.host, []).append(req)

    # 5 OAuthIdentities each POST once to oauth during initialize()
    assert len(by_host["oauth.test"]) == 5
    for host in SOURCE_HOSTS:
        assert len(by_host[host]) == 1, f"expected exactly one call to {host}"
    assert len(by_host["llm.test"]) == 1

    for req in requests:
        if req.url.host == "oauth.test":
            continue
        assert req.headers["fsreqid"] == "req-xyz"
        assert req.headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_build_graph_source_failure_raises_source_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env(monkeypatch)

    synthesis_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal synthesis_calls
        host = request.url.host
        if host == "oauth.test":
            return httpx.Response(200, json={"access_token": "tok"})
        if host == "insights.test":
            return httpx.Response(500, json={"error": "boom"})
        if host == "llm.test":
            synthesis_calls += 1
            return httpx.Response(200, json={})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)

    async with build_graph(transport=transport) as graph:
        with pytest.raises(SourceFetchError) as exc:
            await run(graph, fs_req_id="req-fail")

    assert exc.value.source == "insights"
    assert synthesis_calls == 0
