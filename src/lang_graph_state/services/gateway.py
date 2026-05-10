"""
The sole integration point with the LLM gateway. Configure via env vars; mock this class in tests.
"""
import logging
import os

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class GatewayClient:
    def __init__(self) -> None:
        self._base_url = os.environ["LLM_GATEWAY_BASE_URL"]
        self._model_id = os.environ["LLM_MODEL_ID"]
        # wrap_openai is a no-op when LANGSMITH_API_KEY is unset; it adds free LangSmith tracing when it is.
        self._async_client = wrap_openai(AsyncOpenAI(
            base_url=self._base_url,
            api_key=os.environ.get("LLM_GATEWAY_API_KEY", ""),
        ))
        logger.info("Initialized LLM gateway base_url=%s model=%s", self._base_url, self._model_id)

    async def acomplete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        response = await self._async_client.chat.completions.create(
            model=self._model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
