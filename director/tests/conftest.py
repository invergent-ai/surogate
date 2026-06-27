"""Offline test fixtures: a deterministic worker pool and a stub featurizer.

These let the full label -> SFT -> CMA-ES -> eval pipeline run with zero network calls
and zero cost. Worker "a" is correct on TYPE_A questions, worker "b" on TYPE_B.
"""

from __future__ import annotations

import pytest
import torch

from director.config import WorkerSpec
from director.fugu.inference import attach_worker_ids
from director.fugu.model import Featurizer, SelectionRouter
from director.shared.providers import FakeProvider, WorkerPool
from director.shared.tasks import Dataset, Task
from director.shared.verifiers import REGISTRY


def _is_right(output: str, solution) -> float:
    return 1.0 if "RIGHT" in output else 0.0


REGISTRY.setdefault("fake_right", _is_right)


def typed_answer(model: str, messages, sampling) -> str:
    text = messages[-1]["content"]
    matches = ("worker-a" in model) == ("TYPE_A" in text)
    return f"answer: {'RIGHT' if matches else 'WRONG'}"


class FakeFeaturizer(Featurizer):
    """Maps a prompt to a 2-d one-hot feature by question type (no trainable params)."""

    def __init__(self):
        super().__init__()
        self.d = 2

    def features(self, texts: list[str]) -> torch.Tensor:
        rows = [[1.0, 0.0] if "TYPE_A" in t else [0.0, 1.0] for t in texts]
        return torch.tensor(rows, dtype=torch.float32)


@pytest.fixture
def workers() -> list[WorkerSpec]:
    return [
        WorkerSpec(worker_id="a", model="prov/worker-a"),
        WorkerSpec(worker_id="b", model="prov/worker-b"),
    ]


@pytest.fixture
def pool(workers) -> WorkerPool:
    return WorkerPool(workers, FakeProvider(typed_answer))


@pytest.fixture
def router() -> SelectionRouter:
    r = SelectionRouter(FakeFeaturizer(), num_workers=2)
    attach_worker_ids(r, ["a", "b"])
    return r


def make_typed_tasks(n_per_type: int = 8) -> Dataset:
    tasks = []
    for i in range(n_per_type):
        tasks.append(
            Task(task_id=f"a{i}", prompt=f"TYPE_A question {i}", solution=None, grader="fake_right")
        )
        tasks.append(
            Task(task_id=f"b{i}", prompt=f"TYPE_B question {i}", solution=None, grader="fake_right")
        )
    return Dataset(tasks, name="typed")
