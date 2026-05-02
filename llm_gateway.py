import anthropic

_client = anthropic.Anthropic()
DEFAULT_MODEL = "claude-haiku-4-5"


def complete(prompt: str, system: str = "", model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return response.content[0].text
