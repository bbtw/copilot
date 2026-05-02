import os

from openai import OpenAI


class GatewayClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=os.environ["LLM_GATEWAY_BASE_URL"],
            api_key=os.environ.get("LLM_GATEWAY_API_KEY", "ollama"),
        )
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
