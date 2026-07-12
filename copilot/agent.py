"""The Agent: the system prompt, the solve_plan tool, and the turn loop that
drives them (ADR-0002, CONTEXT.md). The web UI (`.web`) and the eval
Scenario (`.evals.scenario`) are its two frontends: each turn returns a
TurnResult recording every solve_plan execution, so presentation and scoring
live with the frontends, not here.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

from .llm import run_tool_loop
from .model import ModelTooLargeError, solve
from .plan import Infeasible, Objective, Plan, to_summary
from .profile import IncomeKind, from_solve_args
from .taxdata import FilingStatus

SYSTEM_PROMPT = """\
You are the chat interface for a retirement planning optimizer. A deterministic
optimization model — not you — produces every financial number.

How a session works:
1. Learn the user's situation conversationally and fill the solve_plan inputs:
   ages (current, planned retirement, plan horizon), filing status, gross annual
   income, annual savings capacity, annual retirement spending need, balances in
   the three accounts (traditional pre-tax, Roth, taxable brokerage), any Social
   Security or pension (annual amount and start age), optional employer match,
   and optional overrides (expected real return, capital gains rate,
   traditional+Roth contribution limit).
2. Ask which objective to maximize: wealth at retirement, or wealth at the end
   of the plan (terminal wealth). Both are measured after tax.
3. Call solve_plan. The full year-by-year table is shown directly to the
   user by the application — do not reproduce it. You receive a
   summary; narrate the strategy it shows and cite only its numbers.
4. Treat "what if" questions as: adjust the inputs, call solve_plan again,
   compare the summaries.

Hard rules:
- Never compute, estimate, or extrapolate financial figures yourself. Every
  number you state must come from a solve_plan result; you may round it for
  readability but never combine or derive figures.
- All amounts are in today's dollars and the expected return is a real
  (after-inflation) return. Say so when collecting inputs.
- Present results as "optimal under your stated assumptions" — never as
  personal financial advice or a recommendation to act.
- Mention the defaults you applied (5% real return, 15% capital gains rate,
  $24,500 contribution limit) so the user can override them.
- Known simplifications, if asked: federal income tax only; Social Security
  taxed at a flat 85%; taxable account taxed as annual drag on returns (no
  basis tracking); a married couple is pooled into one profile; no retirement-
  account withdrawals before age 60; retirement no later than age 73.
- If the solver reports the plan infeasible, relay its reason and suggest
  which input to revisit.
"""

_STREAM_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": [k.value for k in IncomeKind]},
        "annual_amount": {"type": "number"},
        "start_age": {"type": "integer"},
    },
    "required": ["kind", "annual_amount", "start_age"],
}

SOLVE_TOOL = {
    "type": "function",
    "function": {
        "name": "solve_plan",
        "description": (
            "Solve the lifetime retirement optimization for a profile and an "
            "objective. Returns a summary of the optimal plan; the full "
            "year-by-year table is shown to the user automatically. All dollar "
            "amounts are in today's dollars."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_age": {"type": "integer"},
                "retirement_age": {"type": "integer"},
                "horizon_age": {"type": "integer", "description": "Final plan age, e.g. 90"},
                "filing_status": {"type": "string", "enum": [f.value for f in FilingStatus]},
                "gross_income": {"type": "number", "description": "Gross annual income while working"},
                "savings_capacity": {"type": "number", "description": "Annual amount available to save"},
                "spending_need": {"type": "number", "description": "Annual after-tax spending in retirement"},
                "balance_traditional": {"type": "number"},
                "balance_roth": {"type": "number"},
                "balance_taxable": {"type": "number"},
                "income_streams": {"type": "array", "items": _STREAM_SCHEMA},
                "employer_match": {"type": "number", "description": "Annual employer match, if any"},
                "expected_return": {"type": "number", "description": "Real annual return, default 0.05"},
                "capital_gains_rate": {"type": "number", "description": "Default 0.15"},
                "contribution_limit": {"type": "number", "description": "Traditional+Roth annual cap, default 24500"},
                "objective": {"type": "string", "enum": [o.value for o in Objective]},
            },
            "required": [
                "current_age", "retirement_age", "horizon_age", "filing_status",
                "gross_income", "savings_capacity", "spending_need",
                "balance_traditional", "balance_roth", "balance_taxable",
                "objective",
            ],
        },
    },
}


@dataclass(frozen=True)
class SolveCall:
    """One solve_plan execution: the raw arguments as the LLM passed them
    (None if the JSON didn't parse — what Profile Fidelity scores), the result
    dict the LLM narrates from, and the solved Plan when feasible."""

    args: dict | None
    result: dict
    plan: Plan | None


@dataclass(frozen=True)
class TurnResult:
    reply: str
    solves: tuple[SolveCall, ...]


def _execute_solve(arguments: str) -> SolveCall:
    """Run one solve_plan tool call; failures become the {"error": ...} result
    the LLM sees, never an exception."""
    args: dict | None = None
    try:
        args = json.loads(arguments)
        profile, objective = from_solve_args(args)
        result = solve(profile, objective)
    except json.JSONDecodeError as exc:
        return SolveCall(args=None, result={"error": str(exc)}, plan=None)
    except (ValueError, KeyError, ModelTooLargeError) as exc:
        return SolveCall(args=args, result={"error": str(exc)}, plan=None)
    if isinstance(result, Infeasible):
        return SolveCall(args=args, result={"infeasible": result.reason}, plan=None)
    return SolveCall(args=args, result=to_summary(result), plan=result)


def run_turn(client: OpenAI, model: str, messages: list[dict]) -> TurnResult:
    """Advance the conversation by one user turn via solve_plan tool calls.
    Both the REPL and the eval harness (ADR-0004) drive this same loop."""
    solves: list[SolveCall] = []

    def execute(arguments: str) -> str:
        call = _execute_solve(arguments)
        solves.append(call)
        return json.dumps(call.result)

    reply = run_tool_loop(client, model, messages, tools=[SOLVE_TOOL], execute_tool=execute)
    return TurnResult(reply=reply, solves=tuple(solves))
