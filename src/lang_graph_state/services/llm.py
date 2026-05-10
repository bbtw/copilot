import logging
import os
from urllib.parse import urlparse

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

from lang_graph_state.instrumentation.timing import timed

logger = logging.getLogger(__name__)


class GatewayClient:
    def __init__(self) -> None:
        self._base_url = os.environ["LLM_GATEWAY_BASE_URL"]
        _reject_direct_local_llm_url(self._base_url)
        self._model_provider = os.environ.get("LLM_MODEL_PROVIDER", "")
        self._model_id = os.environ["LLM_MODEL_ID"]
        self._client = wrap_openai(OpenAI(
            base_url=self._base_url,
            api_key=os.environ.get("LLM_GATEWAY_API_KEY", "ollama"),
        ))
        self._async_client = wrap_openai(AsyncOpenAI(
            base_url=self._base_url,
            api_key=os.environ.get("LLM_GATEWAY_API_KEY", "ollama"),
        ))
        logger.info(
            "Initialized LLM gateway base_url=%s model=%s provider=%s",
            self._base_url,
            self._model_id,
            self._model_provider or "default",
        )

    @timed("llm_gateway_complete", finish_attrs=lambda content: {"output_chars": len(content or "")})
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

    @timed("llm_gateway_acomplete", finish_attrs=lambda content: {"output_chars": len(content or "")})
    async def acomplete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        extra = {"model_provider": self._model_provider} if self._model_provider else {}
        response = await self._async_client.chat.completions.create(
            model=self._model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            extra_body=extra,
        )
        return response.choices[0].message.content


def _reject_direct_local_llm_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port == 11434:
        raise ValueError(
            "LLM_GATEWAY_BASE_URL must point at the FastAPI LLM gateway, not Ollama directly. "
            "Use http://127.0.0.1:8001/v1 for local runs."
        )
