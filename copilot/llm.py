"""LLM interaction mechanics: client construction and the two calling patterns
used across the chat agent and the eval harness (ADR-0003). No prompts, tool
schemas, or domain logic live here — callers supply those."""

from collections.abc import Callable

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from .settings import Settings


def llm_client(settings: Settings) -> OpenAI:
    """OpenAI-compatible client for whatever `base_url` points at — the company
    gateway or a local Ollama instance (ADR-0003) — traced via wrap_openai."""
    return wrap_openai(OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key))


def complete(client: OpenAI, model: str, messages: list[dict]) -> str:
    """One plain text completion, no tools."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def run_tool_loop(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    execute_tool: Callable[[str], str],
) -> str:
    """Advance the conversation by one user turn: call the model, execute any
    tool calls via `execute_tool`, and repeat until the assistant replies in
    text. Appends every message to `messages` in place and returns the reply."""
    while True:
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
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
                    "content": execute_tool(tc.function.arguments),
                }
            )
