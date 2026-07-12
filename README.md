# Retirement Planning Optimizer

A chat-driven retirement planner: you describe your situation in plain language,
an LLM fills a typed profile, and a Gurobi linear program finds the tax-optimal
contribution, withdrawal, and Roth-conversion schedule. The LLM never computes a
number — every figure comes from the solver.

See [`docs/guide.html`](docs/guide.html) for setup, running, evals, architecture,
and design decisions (open it in a browser: `open docs/guide.html`).
