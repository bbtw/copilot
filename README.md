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

## Evals

Agentic eval suite (`docs/adr/0004`): a Simulated User improvises from a Fact
Card against the real chat loop; deterministic scorers grade Profile Fidelity
and Number Faithfulness in LangSmith experiments. Fact Cards live in
`copilot/evals/cards/` (the repo is the master; they sync up on each run).

```sh
export LANGSMITH_API_KEY=...
export EVAL_SIM_MODEL=...   # optional: simulated-user model, defaults to LLM_MODEL

uv run copilot-evals --runs 3
```
