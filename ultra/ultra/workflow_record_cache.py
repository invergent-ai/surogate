"""Disk cache for executed Ultra workflow records.

This cache is intentionally one level above provider-completion caching: it
memoizes the terminal result of an already executed task/workflow/worker setup.
It is safe for GRPO because the current Conductor still samples the workflow and
the trainer still uses current-policy logprobs; only deterministic downstream
worker/harness execution is skipped on exact repeats.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import RolloutRecord, TaskSpec, Workflow
from .workers import Sampling

CACHE_VERSION = "ultra-workflow-record-cache-v1"


def _canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def workflow_record_cache_key(
    *,
    task: TaskSpec,
    workflow: Workflow,
    worker_ids: list[str],
    worker_models: dict[str, str],
    worker_harnesses: dict[str, str],
    sampling: Sampling,
    max_steps: int,
    execution_fingerprint: dict[str, Any] | None = None,
) -> str:
    """Return a stable key for a fully specified workflow execution."""

    effective_harnesses = {worker: worker_harnesses.get(worker, task.environment.harness) for worker in worker_ids}
    payload = {
        "version": CACHE_VERSION,
        "task": task.model_dump(mode="json"),
        "workflow": workflow.model_dump(mode="json"),
        "worker_ids": worker_ids,
        "worker_models": {worker: worker_models[worker] for worker in worker_ids},
        "worker_harnesses": effective_harnesses,
        "sampling": sampling.as_dict(),
        "max_steps": max_steps,
        "execution_fingerprint": execution_fingerprint or {},
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class WorkflowRecordCache:
    """Small JSON-file cache for ``RolloutRecord`` objects."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser().resolve() if path else None
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        if self.path is None:
            raise RuntimeError("workflow record cache is disabled")
        return self.path / key[:2] / f"{key}.json"

    def get(self, key: str) -> RolloutRecord | None:
        if self.path is None:
            return None
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return RolloutRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, record: RolloutRecord) -> None:
        if self.path is None:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        tmp.replace(path)
