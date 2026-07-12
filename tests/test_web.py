import json
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import copilot.web as web

ARGS = {
    "current_age": 40,
    "retirement_age": 65,
    "horizon_age": 90,
    "filing_status": "single",
    "gross_income": 150000,
    "savings_capacity": 30000,
    "spending_need": 80000,
    "balance_traditional": 200000,
    "balance_roth": 50000,
    "balance_taxable": 100000,
    "objective": "terminal_wealth",
}


class StubClient:
    """Plays back scripted chat responses in order, ignoring the request."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: next(self._responses))
        )


class FailingClient:
    """Raises on every chat call, standing in for a gateway outage."""

    def __init__(self) -> None:
        def _raise(**kwargs: object) -> None:
            raise RuntimeError("gateway down")

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_raise))


def _text(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _solve_call(arguments: str) -> SimpleNamespace:
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="solve_plan", arguments=arguments),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(autouse=True)
def fresh_state() -> Iterator[None]:
    del web._messages[1:]
    web._transcript.clear()
    web._model = "model"
    yield


def _http(llm: object) -> TestClient:
    web._client = llm
    return TestClient(web.app)


def test_message_turn_returns_plan_and_reply_events() -> None:
    http = _http(StubClient([_solve_call(json.dumps(ARGS)), _text("Narration.")]))
    events = http.post("/api/message", json={"text": "plan my retirement"}).json()["events"]
    plan_event, reply_event = events
    assert plan_event["type"] == "plan"
    assert plan_event["plan"]["objective"] == "terminal_wealth"
    assert plan_event["plan"]["objective_value"] > 0
    assert len(plan_event["plan"]["rows"]) == 50
    assert reply_event == {"type": "assistant", "text": "Narration."}


def test_transcript_replays_the_conversation() -> None:
    http = _http(StubClient([_solve_call(json.dumps(ARGS)), _text("Narration.")]))
    http.post("/api/message", json={"text": "plan my retirement"})
    kinds = [event["type"] for event in http.get("/api/transcript").json()]
    assert kinds == ["user", "plan", "assistant"]


def test_reset_clears_conversation() -> None:
    http = _http(StubClient([_text("Hi!")]))
    http.post("/api/message", json={"text": "hello"})
    http.post("/api/reset")
    assert http.get("/api/transcript").json() == []
    assert [m["role"] for m in web._messages] == ["system"]


def test_gateway_error_reports_and_keeps_conversation_alive() -> None:
    http = _http(FailingClient())
    response = http.post("/api/message", json={"text": "hello"})
    assert response.status_code == 200
    (event,) = response.json()["events"]
    assert event["type"] == "error"
    assert "gateway down" in event["text"]
    assert web._messages[-1] == {"role": "user", "content": "hello"}


def test_index_serves_the_page() -> None:
    response = _http(StubClient([])).get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
