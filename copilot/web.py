"""Local web UI: the consumer frontend to the Agent (`.agent`). Presentation
only — a FastAPI app serving one static page plus JSON endpoints around
run_turn; the prompt, tool schema, and turn loop live with the Agent. State is
one in-memory conversation, the web twin of the old REPL's local variable."""

import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .agent import SYSTEM_PROMPT, run_turn
from .llm import llm_client
from .plan import Plan
from .settings import Settings, sync_langsmith_env

HOST = "127.0.0.1"
PORT = 8765

app = FastAPI()

_client: OpenAI | None = None  # set by main(); tests inject a stub
_model = ""
_lock = threading.Lock()  # one conversation: serialize turns across requests
_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
_transcript: list[dict] = []  # display events: user | assistant | error | plan


class UserMessage(BaseModel):
    text: str


def _plan_event(plan: Plan) -> dict:
    return {
        "type": "plan",
        "plan": {
            "objective": plan.objective.value,
            "objective_value": plan.objective_value,
            "rows": [asdict(row) for row in plan.rows],
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/transcript")
def transcript() -> list[dict]:
    return _transcript


@app.post("/api/reset")
def reset() -> dict:
    with _lock:
        del _messages[1:]
        _transcript.clear()
    return {"ok": True}


@app.post("/api/message")
def message(user: UserMessage) -> dict:
    with _lock:
        _messages.append({"role": "user", "content": user.text})
        _transcript.append({"type": "user", "text": user.text})
        try:
            turn = run_turn(_client, _model, _messages)
        except Exception as exc:  # gateway/network failure: report and keep the conversation alive
            events = [{"type": "error", "text": f"gateway error: {exc}"}]
        else:
            events = [_plan_event(call.plan) for call in turn.solves if call.plan is not None]
            events.append({"type": "assistant", "text": turn.reply})
        _transcript.extend(events)
        return {"events": events}


def main() -> None:
    global _client, _model
    try:
        settings = Settings()
    except ValidationError:
        sys.exit(
            "Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL "
            "(and optionally LANGSMITH_TRACING/LANGSMITH_API_KEY for tracing), "
            "in your shell or a .env file."
        )
    sync_langsmith_env(settings)
    _client = llm_client(settings)
    _model = settings.llm_model
    url = f"http://{HOST}:{PORT}"
    print(f"Retirement planning optimizer — {url} (Ctrl-C to stop)")
    threading.Timer(0.5, webbrowser.open, [url]).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
