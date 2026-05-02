from typing import Protocol

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"


class LLMClient(Protocol):
    # Structural interface: any production client or test fake with this
    # method can be injected into services without inheritance.
    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        ...


class AnthropicLLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        # Provider-specific API shape is isolated here; services only depend on
        # the LLMClient protocol's prompt-completion capability.
        messages = [{"role": "user", "content": prompt}]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text
