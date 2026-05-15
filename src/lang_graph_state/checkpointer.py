from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from lang_graph_state.settings import Settings


def make_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    if settings.checkpointer_backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        if not settings.checkpointer_postgres_url:
            raise ValueError("CHECKPOINTER_POSTGRES_URL must be set when CHECKPOINTER_BACKEND=postgres")
        return AsyncPostgresSaver.from_conn_string(settings.checkpointer_postgres_url)
    return MemorySaver()
