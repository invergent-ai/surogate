"""GRPO rollout collection for the Fugu-Ultra conductor (r5).

The Conductor recipe, adapted to our stack: for each verifiable question,
sample G plans from the served policy at temperature 1.0 (typed contract,
constrained decoding, logprobs recorded), execute each plan through the heavy
orchestrator with CONSTRAINED workers, and grade 0 / 0.5 / 1:

    0    unparseable plan (format condition)
    0.5  well-formed plan whose executed answer is wrong
    1    executed answer correct

Groups sharing a question are the GRPO unit: advantage is computed within the
group, so a group only carries gradient when its rewards DIFFER. The collector
therefore reports the informative-group fraction — the single number that
decides whether GRPO is viable at the current sampling settings before any
trainer time is spent.

Everything is injected (sampler, executor, grader) so the orchestration logic
is offline-testable; the paid pilot wires real components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

REWARD_UNPARSEABLE = 0.0
REWARD_WRONG = 0.5
REWARD_CORRECT = 1.0


class PlanSampler(Protocol):
    """One temperature>0 sample: (steps, meta with raw/logprobs/parse_ok)."""

    def __call__(self, question: str) -> tuple[list, dict]: ...


class PlanExecutor(Protocol):
    """Execute a plan's steps for a question; return the final output text."""

    def __call__(self, question: str, steps: list) -> str: ...


Grader = Callable[[str, Any], bool]  # (final_output, gold) -> correct?


@dataclass
class RolloutCollector:
    """Collects reward groups; grading is dispatched PER QUESTION.

    The r5 pool mixes grader types (math_equal / rlpr_lenient /
    code_exec_stdio) whose gold shapes overlap — a single grader misgrades
    a large slice of any mixed pool, so `collect_question` takes the
    question's own `grader_type` and resolves it through `grader_registry`
    (default: the ultra grading registry). The injected `grader` is the
    fallback for callers whose pool really is single-grader (tests, the
    decision runs).
    """

    sampler: PlanSampler
    executor: PlanExecutor
    grader: Grader
    grader_registry: Callable[[str], Callable[[str, Any], float]] | None = None
    group_size: int = 8
    # >1 runs the group's sample+execute cycles CONCURRENTLY (vLLM batches the
    # plan samples; executions already wave-parallel their leaves). Rollouts
    # within a group are independent by definition, so ordering is free.
    parallelism: int = 1
    log: Callable[[str], None] = print

    def _resolve_grader(self, grader_type: str | None) -> Grader:
        if grader_type is None:
            return self.grader
        registry = self.grader_registry
        if registry is None:
            from ultra.grading.verifiers import get_grader
            registry = get_grader
        graded = registry(grader_type)
        return lambda output, gold: float(graded(output, gold)) >= 1.0

    def _one_rollout(self, question: str, gold: Any,
                     grader: Grader) -> dict[str, Any]:
        try:
            steps, meta = self.sampler(question)
        except Exception as exc:  # noqa: BLE001 — serving hiccup, not policy
            # Same contract as executor failures: reward None keeps the
            # rollout out of the group and the incomplete group re-probes
            # on resume. A transient conductor timeout must never kill a
            # multi-hour probe.
            return {"steps": 0, "raw": "", "logprobs": None,
                    "reward": None, "infra_error": type(exc).__name__}

        def record(**extra: Any) -> dict[str, Any]:
            row = {
                "steps": len(steps),
                "raw": meta.get("raw", ""),
                "logprobs": meta.get("logprobs"),
                **extra,
            }
            # Exact-token ids from the serving stack, when the sampler
            # provides them (see TrainedConductor.sample_plan).
            for key in ("token_ids", "prompt_token_ids"):
                if meta.get(key) is not None:
                    row[key] = meta[key]
            return row

        if not meta.get("parse_ok") or not steps:
            return record(reward=REWARD_UNPARSEABLE, steps=0)
        try:
            output = self.executor(question, steps)
        except Exception as exc:  # noqa: BLE001 — infra failure, not policy
            return record(reward=None,  # excluded from the group, never 0
                          infra_error=type(exc).__name__)
        correct = grader(output, gold)
        return record(reward=REWARD_CORRECT if correct else REWARD_WRONG)

    def collect_question(self, question: str, gold: Any,
                         grader_type: str | None = None) -> dict[str, Any]:
        grader = self._resolve_grader(grader_type)
        if self.parallelism > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
                rollouts = list(pool.map(
                    lambda _: self._one_rollout(question, gold, grader),
                    range(self.group_size),
                ))
        else:
            rollouts = [self._one_rollout(question, gold, grader)
                        for _ in range(self.group_size)]
        rewards = [r["reward"] for r in rollouts if r["reward"] is not None]
        return {
            "question": question,
            "gold": gold,
            "grader_type": grader_type,
            "rollouts": rollouts,
            "rewards": rewards,
            "informative": len(set(rewards)) > 1,
            "infra_errors": sum(1 for r in rollouts if r["reward"] is None),
        }

    def collect(
        self, questions: list[tuple[str, str]], output_path: Path
    ) -> dict[str, Any]:
        informative = 0
        total = 0
        with output_path.open("w") as handle:
            for number, (question, gold) in enumerate(questions, 1):
                group = self.collect_question(question, gold)
                handle.write(json.dumps(group) + "\n")
                total += 1
                informative += group["informative"]
                self.log(
                    f"  [{number}/{len(questions)}] rewards={group['rewards']} "
                    f"informative={group['informative']}"
                )
        stats = {
            "groups": total,
            "informative_groups": informative,
            "informative_fraction": informative / total if total else 0.0,
        }
        self.log(f"informative groups: {informative}/{total}")
        return stats
