# LLM access goes through the company gateway, not a provider SDK

All LLM calls must route through the company's LLM gateway — direct provider SDKs (Anthropic, OpenAI-direct) are not available in this environment. The gateway is OpenAI-compatible, so the client is the standard `openai` Python package pointed at the gateway `base_url`. LangSmith instrumentation (a v1 requirement: every conversation fully traced) is achieved with `langsmith.wrappers.wrap_openai` on the client plus `@traceable` on the Gurobi solve function, giving one trace tree per conversation: model calls, tool calls, and solver runs as nested spans.

## Consequences

- Model choice is whatever the gateway routes; the model name is configuration, not code.
- We assume the gateway passes tool/function calling through. If it turns out not to, the fallback is structured-output prompting (the LLM emits a JSON Profile in-band) — ADR-0002's "LLM translates, solver decides" boundary survives unchanged, only the wire format differs.
- No Anthropic-specific API features (extended thinking config, prompt caching controls) can be relied on.
