from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

DEFAULT_TAGS = ("retirement-plan", "parallel-llm")


def build_invoke_config(
    *,
    thread_id: str | None = None,
    run_name: str = "retirement_plan",
    tags: Sequence[str] = DEFAULT_TAGS,
    metadata: Mapping[str, Any] | None = None,
) -> RunnableConfig:
    """
    Build the RunnableConfig used by graph invocations.

    `thread_id` is the key LangGraph checkpointers use to persist graph state.
    Tags and metadata are picked up by LangSmith when tracing is enabled.
    """
    config: dict[str, Any] = {
        "run_name": run_name,
        "tags": list(tags),
        "metadata": dict(metadata or {}),
    }
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return cast(RunnableConfig, config)

