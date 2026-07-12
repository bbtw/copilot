"""Terminal chat REPL: one frontend to the Agent (`.agent`). Presentation
only — the input loop and Plan table rendering; the prompt, tool schema, and
turn loop live with the Agent."""

import sys

from pydantic import ValidationError

from .agent import SYSTEM_PROMPT, run_turn
from .llm import llm_client
from .plan import render_table
from .settings import Settings, sync_langsmith_env


def main() -> None:
    try:
        settings = Settings()
    except ValidationError:
        sys.exit(
            "Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL "
            "(and optionally LANGSMITH_TRACING/LANGSMITH_API_KEY for tracing), "
            "in your shell or a .env file."
        )
    sync_langsmith_env(settings)
    client = llm_client(settings)
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
            turn = run_turn(client, settings.llm_model, messages)
        except Exception as exc:  # gateway/network failure: report and keep the REPL alive
            print(f"[gateway error: {exc}]")
            continue
        for call in turn.solves:
            if call.plan is not None:
                print(f"\n{render_table(call.plan)}\n")
        print(f"\n{turn.reply}")
