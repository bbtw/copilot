# Agentic evals: LLMs converse, code scores

The eval suite for the chat agent runs full conversations — a Simulated User (an LLM improvising from a Fact Card) against the real chat turn loop through the gateway — but no LLM grades anything. The only two eval dimensions, Profile Fidelity and Number Faithfulness, are scored by deterministic Python functions in the repo: a dict comparison of raw solve_plan arguments against the Fact Card, and a number extractor that checks every figure in the agent's prose against the allowed sources (solve result, Fact Card, plan ages, taxdata defaults) under the significant-digits rounding rule. The same extractor audits the Simulated User's turns; an off-card utterance voids the run rather than counting against the agent. LangSmith `evaluate()` and its experiments UI provide orchestration, repetition (three runs per Scenario), and cross-prompt comparison — but the Fact Cards are mastered in the repo and mirrored up, and the scorers stay in-repo. The Simulated User's model is configuration (`EVAL_SIM_MODEL`, defaulting to `LLM_MODEL`) and its calls go through the same gateway (ADR-0003); every experiment is tagged with the git SHA and a hash of the system prompt, so two experiments are only ever compared anchored to the code and prompt that produced them.

## Considered Options

- **LLM-as-judge** — the conventional choice for agentic evals, and the only way to score soft dimensions like conversation conduct. Rejected for v1: both chosen dimensions are mechanically checkable, and a judge would reintroduce the nondeterminism the suite exists to control for in a prompt-iteration tool.

## Consequences

- Every Fact Card must uniquely determine correct agent behavior — expected solve_plan arguments and allowed numbers follow mechanically from the card. Card families needing interpretation (adversarial phrasing, mid-conversation corrections) are held out of v1 for this reason.
- Judge-dependent dimensions (conversation conduct, What-If discipline) stay out of scope until this decision is consciously revisited; adding a judge later means re-examining the card discipline, not just adding a scorer.
- Scores mean one thing: a fidelity or faithfulness failure is an agent failure by construction (sim-side faults are voided, no-solves are charged to the agent). No transcript needs a human read before a score is believed.
