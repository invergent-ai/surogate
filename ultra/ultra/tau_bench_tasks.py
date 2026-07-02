"""Materialize train-allowed tau-bench retail TaskSpecs.

The installed tau-bench package exposes a retail train split but no airline train
split. Airline remains pool/eval evidence unless we explicitly bless a split.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)

SOURCE_NAME = "tau_bench_retail_train"
SOURCE_VERSION = "sierra-59a200c6"

# High-action retail-train tasks selected from the installed tau-bench train split.
DEFAULT_RETAIL_TRAIN_INDICES = (351, 331, 431, 248, 427, 305, 197, 139, 68, 245, 176, 35)


@dataclass(frozen=True)
class TauBenchTask:
    env_name: str
    task_split: str
    task_index: int
    max_turns: int = 40

    @property
    def task_id(self) -> str:
        return f"tau_bench__{self.env_name}_{self.task_split}_{self.task_index:04d}"


def default_tasks(limit: int | None = None, offset: int = 0) -> list[TauBenchTask]:
    indices = DEFAULT_RETAIL_TRAIN_INDICES[offset:]
    if limit is not None:
        indices = indices[:limit]
    return [TauBenchTask(env_name="retail", task_split="train", task_index=i) for i in indices]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_tasks_train_paths() -> list[Path]:
    roots = [_repo_root() / "director" / ".venv", _repo_root() / ".venv"]
    paths: list[Path] = []
    for root in roots:
        for site_packages in sorted(root.glob("lib/python*/site-packages")):
            paths.append(site_packages / "tau_bench" / "envs" / "retail" / "tasks_train.py")
    return paths


def _retail_train_rows(tasks_train_path: Path | None = None) -> list[dict[str, Any]]:
    paths = [tasks_train_path] if tasks_train_path is not None else _candidate_tasks_train_paths()
    existing = [path for path in paths if path is not None and path.exists()]
    if not existing:
        return []

    module = ast.parse(existing[0].read_text())
    assignment = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and node.targets
            and getattr(node.targets[0], "id", None) == "TASKS_TRAIN"
            and isinstance(node.value, ast.List)
        ),
        None,
    )
    if assignment is None:
        return []

    rows: list[dict[str, Any]] = []
    for index, task_call in enumerate(assignment.value.elts):
        if not isinstance(task_call, ast.Call):
            continue
        instruction = ""
        action_count = 0
        for keyword in task_call.keywords:
            if keyword.arg == "instruction" and isinstance(keyword.value, ast.Constant):
                instruction = str(keyword.value.value)
            if keyword.arg == "actions" and isinstance(keyword.value, ast.List):
                action_count = len(keyword.value.elts)
        rows.append(
            {
                "task_index": index,
                "action_count": action_count,
                "instruction_chars": len(instruction),
            }
        )
    return rows


def high_action_retail_train_indices(
    *,
    limit: int | None = None,
    offset: int = 0,
    tasks_train_path: Path | None = None,
) -> list[int]:
    """Return retail train indices ranked by expected tool/dialogue complexity."""

    rows = _retail_train_rows(tasks_train_path)
    if not rows:
        indices = list(DEFAULT_RETAIL_TRAIN_INDICES)
    else:
        rows = sorted(
            rows,
            key=lambda row: (
                -int(row["action_count"]),
                -int(row["instruction_chars"]),
                int(row["task_index"]),
            ),
        )
        indices = [int(row["task_index"]) for row in rows]
    indices = indices[offset:]
    if limit is not None:
        indices = indices[:limit]
    return indices


def selected_tasks(
    *,
    limit: int | None = None,
    offset: int = 0,
    selection: str = "default",
    tasks_train_path: Path | None = None,
) -> list[TauBenchTask]:
    if selection == "default":
        return default_tasks(limit=limit, offset=offset)
    if selection == "high_action":
        return [
            TauBenchTask(env_name="retail", task_split="train", task_index=index)
            for index in high_action_retail_train_indices(
                limit=limit,
                offset=offset,
                tasks_train_path=tasks_train_path,
            )
        ]
    raise ValueError(f"unknown tau-bench selection {selection!r}")


def task_spec(task: TauBenchTask) -> TaskSpec:
    group = f"{SOURCE_NAME}/{task.env_name}/{task.task_split}/{task.task_index}"
    return TaskSpec(
        task_id=task.task_id,
        capability="tool_dialogue",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref="https://github.com/sierra-research/tau-bench@59a200c6",
        ),
        input=TaskInput(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Run tau-bench {task.env_name} {task.task_split} task "
                        f"{task.task_index} with programmatic reward."
                    ),
                }
            ]
        ),
        environment=EnvironmentSpec(harness="tau_bench", network_policy="model-relay-only", wall_time_seconds=1200),
        grader=GraderSpec(
            type="tau_bench_programmatic",
            expected_answer={
                "env_name": task.env_name,
                "task_split": task.task_split,
                "task_index": task.task_index,
                "user_strategy": "instruction",
                "max_turns": task.max_turns,
            },
        ),
        splitting=SplittingSpec(
            group_id=f"{task.env_name}_{task.task_split}",
            split="grpo_train",
            contamination_group=group,
        ),
        metadata=TaskMetadata(
            domain="tool_dialogue",
            subdomain=f"tau_bench_{task.env_name}",
            tags=["tau_bench", task.env_name, task.task_split, "real_tools", "programmatic_reward"],
            requires_tools=True,
            estimated_worker_calls=task.max_turns,
        ),
    )


def materialize_tau_bench_tasks(
    *,
    out_jsonl: Path,
    report_out: Path | None = None,
    limit: int | None = None,
    offset: int = 0,
    selection: str = "default",
    tasks_train_path: Path | None = None,
) -> dict[str, Any]:
    tasks = selected_tasks(limit=limit, offset=offset, selection=selection, tasks_train_path=tasks_train_path)
    specs = [task_spec(task) for task in tasks]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")
    report = {
        "version": "tau_bench_retail_train_tasks_v1",
        "source": SOURCE_NAME,
        "task_count": len(specs),
        "task_indices": [task.task_index for task in tasks],
        "offset": offset,
        "selection": selection,
        "tasks_train_path": str(tasks_train_path) if tasks_train_path else None,
        "out_jsonl": str(out_jsonl),
        "splits": sorted({spec.splitting.split for spec in specs}),
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
