"""Terminal chat REPL: the LLM translates, the solver decides (ADR-0002).

LLM calls go through the company's OpenAI-compatible gateway (ADR-0003) and are
traced in LangSmith via wrap_openai; solver runs are traced via @traceable.
"""

import json
import os
import sys

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from .model import ModelTooLargeError, solve
from .plan import Infeasible, Objective, render_table, to_summary
from .profile import IncomeKind, IncomeStream, Profile
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
3. Call solve_plan. The full year-by-year table is printed directly to the
   user's terminal by the application — do not reproduce it. You receive a
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

_OPTIONAL_FIELDS = (
    "employer_match", "expected_return", "capital_gains_rate", "contribution_limit",
)


def _profile_from_args(args: dict) -> tuple[Profile, Objective]:
    objective = Objective(args["objective"])
    kwargs = {
        "current_age": int(args["current_age"]),
        "retirement_age": int(args["retirement_age"]),
        "horizon_age": int(args["horizon_age"]),
        "filing_status": FilingStatus(args["filing_status"]),
        "gross_income": float(args["gross_income"]),
        "savings_capacity": float(args["savings_capacity"]),
        "spending_need": float(args["spending_need"]),
        "balance_traditional": float(args["balance_traditional"]),
        "balance_roth": float(args["balance_roth"]),
        "balance_taxable": float(args["balance_taxable"]),
        "income_streams": tuple(
            IncomeStream(
                kind=IncomeKind(s["kind"]),
                annual_amount=float(s["annual_amount"]),
                start_age=int(s["start_age"]),
            )
            for s in args.get("income_streams", [])
        ),
    }
    for field in _OPTIONAL_FIELDS:
        if args.get(field) is not None:
            kwargs[field] = float(args[field])
    return Profile(**kwargs), objective


def _run_tool_call(arguments: str) -> str:
    try:
        profile, objective = _profile_from_args(json.loads(arguments))
        result = solve(profile, objective)
    except (ValueError, KeyError, ModelTooLargeError) as exc:
        return json.dumps({"error": str(exc)})
    if isinstance(result, Infeasible):
        return json.dumps({"infeasible": result.reason})
    print(f"\n{render_table(result)}\n")
    return json.dumps(to_summary(result))


def gateway_client() -> OpenAI:
    """OpenAI client for the company gateway (ADR-0003), traced via wrap_openai."""
    return wrap_openai(
        OpenAI(
            base_url=os.environ["LLM_GATEWAY_BASE_URL"],
            api_key=os.environ["LLM_GATEWAY_API_KEY"],
        )
    )


def run_turn(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Advance the conversation by one user turn: call the model, execute any
    solve_plan calls, and repeat until the assistant replies in text. Appends
    every message to `messages` in place and returns the reply. Both the REPL
    and the eval harness (ADR-0004) drive this same loop."""
    while True:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=[SOLVE_TOOL]
        )
        msg = response.choices[0].message
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(entry)
        if not msg.tool_calls:
            return msg.content or ""
        for tc in msg.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _run_tool_call(tc.function.arguments),
                }
            )


def main() -> None:
    base_url = os.environ.get("LLM_GATEWAY_BASE_URL")
    api_key = os.environ.get("LLM_GATEWAY_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base_url and api_key and model):
        sys.exit(
            "Set LLM_GATEWAY_BASE_URL, LLM_GATEWAY_API_KEY, and LLM_MODEL "
            "(and optionally LANGSMITH_TRACING/LANGSMITH_API_KEY for tracing)."
        )
    client = gateway_client()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Retirement planning optimizer — describe your situation ('quit' to exit).")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        messages.append({"role": "user", "content": user_input})
        try:
            reply = run_turn(client, model, messages)
        except Exception as exc:  # gateway/network failure: report and keep the REPL alive
            print(f"[gateway error: {exc}]")
            continue
        print(f"\n{reply}")
