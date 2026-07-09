# Retirement Planning Optimizer

A chat-driven retirement planner: you describe your situation in plain language,
an LLM fills a typed profile, and a Gurobi linear program finds the tax-optimal
contribution, withdrawal, and Roth-conversion schedule. The LLM never computes a
number — every figure comes from the solver (`docs/adr/0002`).

Domain language lives in [`CONTEXT.md`](CONTEXT.md); design decisions in
[`docs/adr/`](docs/adr/).

## Run

```sh
export LLM_GATEWAY_BASE_URL=...   # company OpenAI-compatible gateway
export LLM_GATEWAY_API_KEY=...
export LLM_MODEL=...              # model name your gateway routes
export LANGSMITH_TRACING=true     # optional: full tracing
export LANGSMITH_API_KEY=...

uv run copilot
```

Gurobi runs on the free pip license (no license file needed); the model is sized
to stay under its 2,000-variable/constraint limit.

## Test

```sh
uv run pytest
```
