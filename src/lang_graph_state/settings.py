from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OAuth token server
    oauth_token_url: str = Field(..., alias="OAUTH_TOKEN_URL")

    # Customer Profile API
    customer_profile_url: str = Field(..., alias="CUSTOMER_PROFILE_URL")
    customer_profile_user: str = Field(..., alias="CUSTOMER_PROFILE_USER")
    customer_profile_pass: str = Field(..., alias="CUSTOMER_PROFILE_PASS")

    # Insights API
    insights_url: str = Field(..., alias="INSIGHTS_URL")
    insights_user: str = Field(..., alias="INSIGHTS_USER")
    insights_pass: str = Field(..., alias="INSIGHTS_PASS")

    # Optimizer API
    optimizer_url: str = Field(..., alias="OPTIMIZER_URL")
    optimizer_user: str = Field(..., alias="OPTIMIZER_USER")
    optimizer_pass: str = Field(..., alias="OPTIMIZER_PASS")

    # Monte Carlo API
    monte_carlo_url: str = Field(..., alias="MONTE_CARLO_URL")
    monte_carlo_user: str = Field(..., alias="MONTE_CARLO_USER")
    monte_carlo_pass: str = Field(..., alias="MONTE_CARLO_PASS")

    # LLM Gateway
    llm_gateway_url: str = Field(..., alias="LLM_GATEWAY_BASE_URL")
    llm_gateway_user: str = Field(..., alias="LLM_GATEWAY_USER")
    llm_gateway_pass: str = Field(..., alias="LLM_GATEWAY_PASS")
    llm_model_provider: str = Field(..., alias="LLM_MODEL_PROVIDER")
    llm_model_id: str = Field(..., alias="LLM_MODEL_ID")

    # Checkpointer
    checkpointer_backend: str = Field("memory", alias="CHECKPOINTER_BACKEND")
    checkpointer_postgres_url: str | None = Field(None, alias="CHECKPOINTER_POSTGRES_URL")

    # Observability
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field("", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("", alias="LANGSMITH_PROJECT")
