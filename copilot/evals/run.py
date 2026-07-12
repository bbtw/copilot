"""Eval runner: mirror Fact Cards to LangSmith, run the suite via evaluate() (ADR-0004).

Two suite scores come out — the Profile Fidelity rate and the Number
Faithfulness rate — never blended. A Voided Run (Simulated User went off-card)
is retried and reported separately; it never counts against the agent.
"""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import asdict

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from pydantic import ValidationError

from ..agent import SYSTEM_PROMPT
from ..llm import llm_client
from ..settings import Settings, sync_langsmith_env
from .cards import FactCard, load_cards
from .scenario import TURN_CAP, run_scenario
from .scoring import audit_sim, faithfulness_sources, number_faithfulness, profile_fidelity

DATASET_NAME = "copilot-agentic-evals"
MAX_VOID_RETRIES = 2


def _card_from(inputs: dict) -> FactCard:
    return FactCard(
        name=inputs["name"],
        family=inputs["family"],
        facts=inputs["facts"],
        expected_args=inputs["expected_args"],
    )


def sync_dataset(client: Client, cards: list[FactCard]) -> None:
    """Upsert the repo-mastered Fact Cards into the LangSmith dataset and delete
    examples for cards that no longer exist — the repo is the master copy."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Fact Cards mirrored from copilot/evals/cards; the repo is the master.",
        )
    existing = {
        example.inputs.get("name"): example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    for card in cards:
        inputs = asdict(card)
        current = existing.pop(card.name, None)
        if current is None:
            client.create_example(inputs=inputs, dataset_id=dataset.id)
        elif current.inputs != inputs:
            client.update_example(current.id, inputs=inputs)
    for stale in existing.values():
        client.delete_example(stale.id)


def _fidelity_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    if outputs.get("voided"):
        return {"key": "profile_fidelity", "score": None, "comment": "Voided Run"}
    score = profile_fidelity(outputs.get("solve_args"), _card_from(example.inputs))
    return {"key": "profile_fidelity", "score": int(score.passed), "comment": "; ".join(score.reasons)}


def _faithfulness_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    if outputs.get("voided"):
        return {"key": "number_faithfulness", "score": None, "comment": "Voided Run"}
    sources = faithfulness_sources(_card_from(example.inputs), outputs.get("solve_result"))
    score = number_faithfulness(outputs.get("agent_texts", []), sources)
    return {"key": "number_faithfulness", "score": int(score.passed), "comment": "; ".join(score.reasons)}


def _voided_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    return {
        "key": "voided_run",
        "score": int(bool(outputs.get("voided"))),
        "comment": "; ".join(outputs.get("void_reasons", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic eval suite (ADR-0004).")
    parser.add_argument("--runs", type=int, default=3, help="runs per Scenario (default 3)")
    args = parser.parse_args()

    try:
        settings = Settings()
    except ValidationError:
        sys.exit(
            "Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL "
            "(plus LANGSMITH_API_KEY; EVAL_SIM_MODEL optionally overrides the sim model), "
            "in your shell or a .env file."
        )
    if not settings.langsmith_api_key:
        sys.exit("Set LANGSMITH_API_KEY to run evals.")
    sync_langsmith_env(settings)
    agent_model = settings.llm_model
    sim_model = settings.eval_sim_model or agent_model
    agent_client = llm_client(settings)
    sim_client = llm_client(settings)

    cards = load_cards()
    ls_client = Client()
    sync_dataset(ls_client, cards)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]

    def target(inputs: dict) -> dict:
        card = _card_from(inputs)
        outputs: dict = {}
        for attempt in range(1 + MAX_VOID_RETRIES):
            run = run_scenario(card, agent_client, agent_model, sim_client, sim_model)
            audit = audit_sim(run.sim_texts, card)
            outputs = {
                "voided": not audit.passed,
                "void_reasons": list(audit.reasons),
                "void_attempts": attempt,
                "solve_args": run.solve.args if run.solve is not None else None,
                "solve_result": run.solve.result if run.solve is not None else None,
                "agent_texts": list(run.agent_texts),
                "sim_texts": list(run.sim_texts),
            }
            if audit.passed:
                break
        return outputs

    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[_fidelity_evaluator, _faithfulness_evaluator, _voided_evaluator],
        num_repetitions=args.runs,
        experiment_prefix=f"copilot-evals-{git_sha[:7]}",
        metadata={
            "git_sha": git_sha,
            "system_prompt_sha256": prompt_hash,
            "agent_model": agent_model,
            "sim_model": sim_model,
            "turn_cap": TURN_CAP,
        },
        client=ls_client,
    )
