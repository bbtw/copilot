"""Drive one Scenario: a Simulated User against the real chat loop (ADR-0004).

The agent side is `chat.run_turn` — the same code the REPL runs. A Scenario
ends at the first solve_plan call plus the agent's narration turn, or at the
turn cap without a solve ("no-solve", charged to the agent).
"""

import json
from dataclasses import dataclass

from openai import OpenAI

from ..chat import SYSTEM_PROMPT, run_turn
from .cards import FactCard

TURN_CAP = 12

SIM_PROMPT = """\
You are role-playing a person chatting with their retirement planning
assistant. Everything about your finances is on the fact sheet below.

Rules:
- Never state a number that is not on the fact sheet — no examples, no
  guesses, and never repeat a number the assistant said.
- If asked about something the sheet doesn't cover, say you don't have one
  or tell the assistant to use whatever it normally assumes — without
  naming any figure.
- Speak casually and volunteer a few facts per message, like a real person.
- Your first message should say what you want (the sheet's goal) and start
  describing your situation.

Fact sheet:
{facts}
"""

_KICKOFF = "(You have just opened the chat. Send your first message.)"


@dataclass
class ScenarioRun:
    messages: list[dict]
    solve_args: dict | None
    solve_result: dict | None
    agent_texts: tuple[str, ...]
    sim_texts: tuple[str, ...]


def _completion_text(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def _tool_result(messages: list[dict], start: int, tool_call_id: str) -> dict | None:
    for entry in messages[start:]:
        if entry.get("role") == "tool" and entry.get("tool_call_id") == tool_call_id:
            return json.loads(entry["content"])
    return None


def _first_solve(messages: list[dict], start: int) -> tuple[dict | None, dict | None] | None:
    for i in range(start, len(messages)):
        entry = messages[i]
        if entry["role"] != "assistant":
            continue
        for call in entry.get("tool_calls", []):
            if call["function"]["name"] != "solve_plan":
                continue
            try:
                args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                args = None
            return args, _tool_result(messages, i + 1, call["id"])
    return None


def _finish(
    messages: list[dict], solve_args: dict | None, solve_result: dict | None
) -> ScenarioRun:
    return ScenarioRun(
        messages=messages,
        solve_args=solve_args,
        solve_result=solve_result,
        agent_texts=tuple(
            m["content"] for m in messages if m["role"] == "assistant" and m["content"]
        ),
        sim_texts=tuple(m["content"] for m in messages if m["role"] == "user"),
    )


def run_scenario(
    card: FactCard,
    agent_client: OpenAI,
    agent_model: str,
    sim_client: OpenAI,
    sim_model: str,
    turn_cap: int = TURN_CAP,
) -> ScenarioRun:
    """Run one Scenario conversation and capture the first solve, if any."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    sim_messages: list[dict] = [
        {"role": "system", "content": SIM_PROMPT.format(facts=card.facts)},
        {"role": "user", "content": _KICKOFF},
    ]
    for _ in range(turn_cap):
        sim_text = _completion_text(sim_client, sim_model, sim_messages)
        sim_messages.append({"role": "assistant", "content": sim_text})
        messages.append({"role": "user", "content": sim_text})
        before = len(messages)
        reply = run_turn(agent_client, agent_model, messages)
        solve = _first_solve(messages, before)
        if solve is not None:
            return _finish(messages, *solve)
        sim_messages.append({"role": "user", "content": reply})
    return _finish(messages, None, None)
