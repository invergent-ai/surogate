"""Small artifact helpers for repo-harness trace capture."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..schemas import TaskSpec


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:160] or "artifact"


def artifact_ref(path: Path) -> str:
    return str(path.resolve())


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return artifact_ref(path)


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return artifact_ref(path)


def write_repo_state(path: Path, task: TaskSpec, instance: dict[str, Any]) -> str:
    repo = task.input.repo
    return write_json(
        path,
        {
            "task_id": task.task_id,
            "source_name": task.source.name,
            "repo": repo.model_dump(mode="json") if repo is not None else None,
            "image_name": instance.get("image_name"),
            "instance_id": instance.get("instance_id"),
            "testbed": instance.get("testbed"),
            "tests_dir": instance.get("tests_dir"),
            "task_dir": instance.get("task_dir"),
        },
    )


def copy_workspace(src: Path, dst: Path) -> str:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(src, dst, ignore=ignore)
    return artifact_ref(dst)
