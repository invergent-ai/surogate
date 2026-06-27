"""Task and dataset abstractions.

A ``Task`` is a verifiable unit of work: a prompt, a ground-truth solution, and the
name of the grader that scores a worker's answer against it. Single-step tasks feed
the SFT (soft-target) stage; end-to-end / multi-turn tasks feed sep-CMA-ES.

Dataset loaders pull from HuggingFace ``datasets`` and are intentionally thin; tests
build ``Task`` lists directly so they need no network.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from .types import Message

TaskKind = Literal["single_step", "end_to_end", "multi_turn"]


@dataclass
class Task:
    task_id: str
    prompt: str
    solution: Any
    grader: str
    kind: TaskKind = "single_step"
    system: str | None = None
    metadata: dict = field(default_factory=dict)

    def messages(self) -> list[Message]:
        msgs: list[Message] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.prompt})
        return msgs


class Dataset:
    """A simple in-memory, iterable collection of tasks."""

    def __init__(self, tasks: Iterable[Task], name: str = "dataset"):
        self._tasks = list(tasks)
        self.name = name

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, i: int) -> Task:
        return self._tasks[i]


# ---------------------------------------------------------------------------
# HuggingFace loaders (used for live runs; not imported by offline tests)
# ---------------------------------------------------------------------------

_MATH_SYS = "Solve the problem. Put your final answer in \\boxed{}."
_MC_SYS = "Answer the multiple-choice question. End with 'Answer: X' where X is the letter."


def _hf(path: str, split: str, name: str | None = None):
    from datasets import load_dataset

    return load_dataset(path, name, split=split) if name else load_dataset(path, split=split)


def load_gsm8k(split: str = "test", limit: int | None = None) -> Dataset:
    ds = _hf("openai/gsm8k", split, name="main")
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        tasks.append(
            Task(
                task_id=f"gsm8k-{split}-{i}",
                prompt=row["question"],
                solution=row["answer"],
                grader="gsm8k_exact",
                system="Solve the problem. End your answer with '#### <number>'.",
            )
        )
    return Dataset(tasks, name="gsm8k")


def load_math(split: str = "test", limit: int | None = None) -> Dataset:
    ds = _hf("hendrycks/competition_math", split)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        tasks.append(
            Task(
                task_id=f"math-{split}-{i}",
                prompt=row["problem"],
                solution=row["solution"],
                grader="math_equal",
                system=_MATH_SYS,
            )
        )
    return Dataset(tasks, name="math")


def load_humaneval(limit: int | None = None) -> Dataset:
    ds = _hf("openai/openai_humaneval", "test")
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        tasks.append(
            Task(
                task_id=row["task_id"],
                prompt=row["prompt"],
                solution={"test": row["test"], "entry_point": row["entry_point"]},
                grader="code_exec",
                system="Complete the Python function. Return only the full function.",
            )
        )
    return Dataset(tasks, name="humaneval")


def load_gpqa(split: str = "test", limit: int | None = None) -> Dataset:
    """GPQA-Diamond (ungated MC mirror). Each row's ``problem`` already embeds the four
    options and a "\\boxed{letter}" instruction; ``solution`` is e.g. "\\boxed{D}"."""
    from .verifiers import extract_boxed

    ds = _hf("hendrydong/gpqa_diamond_mc", split)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        letter = (extract_boxed(row["solution"]) or str(row["solution"])).strip()[:1].upper()
        tasks.append(
            Task(
                task_id=f"gpqa-{split}-{i}",
                prompt=row["problem"],
                solution=letter,
                grader="mc_letter",
                system="Answer the question. Put the final answer letter in \\boxed{}.",
                metadata={"domain": row.get("domain")},
            )
        )
    return Dataset(tasks, name="gpqa")


LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "humaneval": load_humaneval,
    "gpqa": load_gpqa,
}
