import os

from langchain_openai import ChatOpenAI
from langsmith.wrappers import wrap_openai
from openai import OpenAI


class GatewayClient:
    def __init__(self) -> None:
        self._client = wrap_openai(OpenAI(
            base_url=os.environ["LLM_GATEWAY_BASE_URL"],
            api_key=os.environ.get("LLM_GATEWAY_API_KEY", "ollama"),
        ))
        self._model_provider = os.environ.get("LLM_MODEL_PROVIDER", "")
        self._model_id = os.environ["LLM_MODEL_ID"]

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        extra = {"model_provider": self._model_provider} if self._model_provider else {}
        response = self._client.chat.completions.create(
            model=self._model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            extra_body=extra,
        )
        return response.choices[0].message.content


def build_chat_model(max_tokens: int = 2048) -> ChatOpenAI:
    """Return a ChatOpenAI pointed at the same gateway as GatewayClient, for use with create_react_agent."""
    model_provider = os.environ.get("LLM_MODEL_PROVIDER", "")
    model_kwargs = {"extra_body": {"model_provider": model_provider}} if model_provider else {}
    return ChatOpenAI(
        base_url=os.environ["LLM_GATEWAY_BASE_URL"],
        api_key=os.environ.get("LLM_GATEWAY_API_KEY", "ollama"),
        model=os.environ["LLM_MODEL_ID"],
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )
