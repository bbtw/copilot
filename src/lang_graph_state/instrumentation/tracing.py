from typing import Any, cast

from langchain_core.runnables import RunnableConfig


def build_invoke_config(*, thread_id: str | None = None) -> RunnableConfig:
    config: dict[str, Any] = {}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return cast(RunnableConfig, config)
