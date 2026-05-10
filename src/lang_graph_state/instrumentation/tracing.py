"""
Builds the LangGraph invocation config. The only required key is configurable.thread_id,
which scopes checkpointed state to a specific thread.
"""
from typing import Any, cast

from langchain_core.runnables import RunnableConfig


def build_invoke_config(*, thread_id: str | None = None) -> RunnableConfig:
    config: dict[str, Any] = {}
    if thread_id is not None:
        # "configurable.thread_id" is the key LangGraph checkpointers use to scope persisted state to a thread.
        config["configurable"] = {"thread_id": thread_id}
    return cast(RunnableConfig, config)
