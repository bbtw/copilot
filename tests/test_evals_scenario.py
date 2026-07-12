import json
from types import SimpleNamespace

from copilot.evals.cards import load_cards
from copilot.evals.scenario import run_scenario

CARD = next(card for card in load_cards() if card.name == "mid-career-single")


class StubClient:
    """Plays back scripted chat responses in order, ignoring the request."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: next(self._responses))
        )


def _text(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _solve_call(args: dict) -> SimpleNamespace:
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="solve_plan", arguments=json.dumps(args)),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_scenario_ends_at_first_solve_plus_narration() -> None:
    sim = StubClient([_text("Hi, I'm 40 and want to plan to 90."), _text("That's everything.")])
    agent = StubClient(
        [
            _text("Got it — what's your income?"),
            _solve_call(CARD.expected_args),
            _text("Done: here is your plan."),
        ]
    )
    run = run_scenario(CARD, agent, "agent-model", sim, "sim-model", turn_cap=5)
    assert run.solve is not None
    assert run.solve.args == CARD.expected_args
    assert "objective_value_after_tax" in run.solve.result
    assert run.agent_texts[-1] == "Done: here is your plan."
    assert len(run.sim_texts) == 2


def test_scenario_hits_turn_cap_as_no_solve() -> None:
    sim = StubClient([_text(f"sim message {i}") for i in range(3)])
    agent = StubClient([_text(f"agent question {i}") for i in range(3)])
    run = run_scenario(CARD, agent, "agent-model", sim, "sim-model", turn_cap=3)
    assert run.solve is None
    assert len(run.sim_texts) == 3
