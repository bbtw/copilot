"""Drive one Scenario: a Simulated User against the real Agent turn loop (ADR-0004).

The agent side is `agent.run_turn` — the same code the REPL runs. A Scenario
ends at the first solve_plan execution plus the Agent's narration turn, or at
the turn cap without a solve ("no-solve", charged to the Agent).
"""

from dataclasses import dataclass

from openai import OpenAI

from ..agent import SYSTEM_PROMPT, SolveCall, run_turn
from ..llm import complete
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
    solve: SolveCall | None
    agent_texts: tuple[str, ...]
    sim_texts: tuple[str, ...]


def _finish(messages: list[dict], solve: SolveCall | None) -> ScenarioRun:
    return ScenarioRun(
        messages=messages,
        solve=solve,
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
        sim_text = complete(sim_client, sim_model, sim_messages)
        sim_messages.append({"role": "assistant", "content": sim_text})
        messages.append({"role": "user", "content": sim_text})
        turn = run_turn(agent_client, agent_model, messages)
        if turn.solves:
            return _finish(messages, turn.solves[0])
        sim_messages.append({"role": "user", "content": turn.reply})
    return _finish(messages, None)
