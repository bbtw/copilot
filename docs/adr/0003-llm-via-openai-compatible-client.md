# LLM access goes through an OpenAI-compatible client, not a provider SDK

All LLM calls use the standard `openai` Python package pointed at a configurable `base_url` — never a provider SDK (Anthropic, OpenAI-direct), since direct provider SDKs aren't available in the production environment. In production, `base_url`/`api_key`/`model` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) point at the company's LLM gateway, which is itself OpenAI-compatible. For local development and fast eval iteration, the same three env vars point at a local Ollama instance serving the OpenAI-compatible API instead — no code branch, `llm_client()` doesn't know or care which target it's talking to. LangSmith instrumentation (a v1 requirement: every conversation fully traced) is achieved with `langsmith.wrappers.wrap_openai` on the client plus `@traceable` on the Gurobi solve function, giving one trace tree per conversation: model calls, tool calls, and solver runs as nested spans — verified to no-op cleanly with no LangSmith credentials set, so local/Ollama runs don't require a LangSmith account.

## Consequences

- Model choice is whatever the target routes; the model name is configuration, not code.
- We assume the target passes tool/function calling through. If it turns out not to, the fallback is structured-output prompting (the LLM emits a JSON Profile in-band) — ADR-0002's "LLM translates, solver decides" boundary survives unchanged, only the wire format differs. This is also why the local Ollama model must be chosen for reliable tool-calling support (e.g. `llama3.1`), not just any locally-available model.
- No Anthropic-specific API features (extended thinking config, prompt caching controls) can be relied on.
- The eval suite's `--local` mode (ADR-0004) reuses this same client against Ollama and skips LangSmith dataset sync/experiment logging, since local runs are for fast dev iteration, not tracked benchmarking.
