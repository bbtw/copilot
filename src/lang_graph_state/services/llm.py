import os
import logging
from time import perf_counter
from urllib.parse import urlparse

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

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

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        extra = {"model_provider": self._model_provider} if self._model_provider else {}
        logger.info(
            "LLM gateway completion start model=%s base_url=%s system_chars=%s prompt_chars=%s max_tokens=%s",
            self._model_id,
            self._base_url,
            len(system),
            len(prompt),
            max_tokens,
        )
        started = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                extra_body=extra,
            )
        except Exception:
            logger.exception("LLM gateway completion failed after %.2fs", perf_counter() - started)
            raise
        content = response.choices[0].message.content
        logger.info(
            "LLM gateway completion finish elapsed=%.2fs output_chars=%s",
            perf_counter() - started,
            len(content or ""),
        )
        return content

    async def acomplete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        extra = {"model_provider": self._model_provider} if self._model_provider else {}
        logger.info(
            "LLM gateway async completion start model=%s base_url=%s system_chars=%s prompt_chars=%s max_tokens=%s",
            self._model_id,
            self._base_url,
            len(system),
            len(prompt),
            max_tokens,
        )
        started = perf_counter()
        try:
            response = await self._async_client.chat.completions.create(
                model=self._model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                extra_body=extra,
            )
        except Exception:
            logger.exception("LLM gateway async completion failed after %.2fs", perf_counter() - started)
            raise
        content = response.choices[0].message.content
        logger.info(
            "LLM gateway async completion finish elapsed=%.2fs output_chars=%s",
            perf_counter() - started,
            len(content or ""),
        )
        return content


def _reject_direct_local_llm_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port == 11434:
        raise ValueError(
            "LLM_GATEWAY_BASE_URL must point at the FastAPI LLM gateway, not Ollama directly. "
            "Use http://127.0.0.1:8001/v1 for local runs."
        )
