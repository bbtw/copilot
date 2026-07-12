"""App configuration: one typed source of truth, read from the process
environment or a `.env` file in the working directory."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_gateway_base_url: str
    llm_gateway_api_key: str
    llm_model: str
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    eval_sim_model: str | None = None


def sync_langsmith_env(settings: Settings) -> None:
    """The langsmith SDK reads LANGSMITH_TRACING/LANGSMITH_API_KEY from the
    process environment itself, so values that came only from `.env` must be
    copied back before any traced call is made."""
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
