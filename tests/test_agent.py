import json
from types import SimpleNamespace

from copilot.agent import TurnResult, run_turn

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


def _run(arguments: str) -> tuple[TurnResult, list[dict]]:
    messages: list[dict] = [{"role": "system", "content": "test"}]
    client = StubClient([_solve_call(arguments), _text("Narration.")])
    return run_turn(client, "model", messages), messages


def test_feasible_solve_records_args_result_and_plan() -> None:
    turn, messages = _run(json.dumps(ARGS))
    assert turn.reply == "Narration."
    (call,) = turn.solves
    assert call.args == ARGS
    assert call.result["objective_value_after_tax"] > 0
    assert call.plan is not None and len(call.plan.rows) == 50
    tool_entries = [m for m in messages if m["role"] == "tool"]
    assert json.loads(tool_entries[0]["content"]) == call.result


def test_invalid_profile_records_error_without_plan() -> None:
    args = dict(ARGS, retirement_age=80)  # v1 caps retirement at the RMD age
    turn, _ = _run(json.dumps(args))
    (call,) = turn.solves
    assert call.args == args
    assert "error" in call.result
    assert call.plan is None


def test_infeasible_solve_records_reason() -> None:
    args = dict(
        ARGS,
        current_age=60,
        retirement_age=60,
        gross_income=0,
        savings_capacity=0,
        spending_need=500000,
        balance_traditional=50000,
        balance_roth=25000,
        balance_taxable=25000,
    )
    turn, _ = _run(json.dumps(args))
    (call,) = turn.solves
    assert "infeasible" in call.result
    assert call.plan is None


def test_unparseable_arguments_record_no_args() -> None:
    turn, _ = _run("{not json")
    (call,) = turn.solves
    assert call.args is None
    assert "error" in call.result
    assert call.plan is None


def test_plain_text_turn_has_no_solves() -> None:
    messages: list[dict] = [{"role": "system", "content": "test"}]
    turn = run_turn(StubClient([_text("Hi!")]), "model", messages)
    assert turn.reply == "Hi!"
    assert turn.solves == ()
