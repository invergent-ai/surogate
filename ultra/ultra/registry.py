"""Immutable TaskSpec registry (ultra-data2 §1, §11).

The single gate every task passes before it can be sampled: it must carry a
registered ``SourceManifest``, its policy must permit its split, and it must not
collide with an already-seen task on the dedup key. The registry rejects anything
that fails — silently keeping malformed tasks was a prior-implementation failure mode.
"""

from __future__ import annotations

from collections.abc import Iterator

from .policy import policy_allows_split
from .schemas import SourceManifest, Split, TaskSpec
from .splits import dedup_key


class RegistryError(ValueError):
    pass


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        self._manifests: dict[str, SourceManifest] = {}
        self._seen_dedup: dict[str, str] = {}  # dedup_key -> task_id

    def register_manifest(self, manifest: SourceManifest) -> None:
        self._manifests[manifest.source_name] = manifest

    def add(self, task: TaskSpec) -> None:
        if task.source.name not in self._manifests:
            raise RegistryError(
                f"no SourceManifest registered for source {task.source.name!r} "
                f"(task {task.task_id})"
            )
        if not policy_allows_split(task.source.policy, task.splitting.split):
            raise RegistryError(
                f"policy {task.source.policy!r} forbids split {task.splitting.split!r} "
                f"(task {task.task_id})"
            )
        if task.task_id in self._tasks:
            raise RegistryError(f"task_id {task.task_id!r} already registered")
        key = dedup_key(task)
        if key in self._seen_dedup:
            raise RegistryError(
                f"task {task.task_id!r} duplicates {self._seen_dedup[key]!r} (dedup {key})"
            )
        self._seen_dedup[key] = task.task_id
        self._tasks[task.task_id] = task

    def add_many(self, tasks: "Iterator[TaskSpec] | list[TaskSpec]") -> int:
        n = 0
        for t in tasks:
            self.add(t)
            n += 1
        return n

    def by_split(self, split: Split) -> list[TaskSpec]:
        return [t for t in self._tasks.values() if t.splitting.split == split]

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[TaskSpec]:
        return iter(self._tasks.values())

    def __getitem__(self, task_id: str) -> TaskSpec:
        return self._tasks[task_id]
