# LLM translates, solver decides

The chat interface is an LLM; the retirement math is a fixed, hand-written Gurobi LP formulation. The LLM's job is translation in both directions — natural language → typed Profile on the way in, solver output → narrative on the way out — connected by tool use. Every financial number the User sees originates in solver output: the LLM never computes, estimates, or extrapolates a figure itself, and it never writes or modifies optimization code per conversation.

> **Amended 2026-07-09**: originally this rule also forbade rounding. When the eval suite pinned Number Faithfulness to a deterministic check, we allowed rounding a solver figure to displayed significant digits ("$1.2M" for 1,203,456) so narration can sound human — computing, estimating, and extrapolating remain forbidden.

## Considered Options

- **LLM generates/edits the optimization model per conversation** — maximally flexible, but a financial answer produced by a formulation written fresh each session can't be tested or trusted, and errors are silent.
- **LLM does its own arithmetic when explaining** — invites hallucinated numbers into a domain where a wrong figure is worse than no figure.

## Consequences

- The formulation is testable in isolation (golden profiles → expected plans) independent of any LLM behavior.
- New model capabilities require changing the formulation code — the chat can't "unlock" behavior the LP doesn't have. This is deliberate.
- What-if questions ("what if I retire at 62?") are implemented as Profile mutation + re-solve, never as LLM extrapolation from a previous answer.
