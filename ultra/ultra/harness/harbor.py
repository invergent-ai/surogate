"""Harbor-backed terminal sandbox execution.

This harness lets Ultra route ``terminal_sandbox`` steps to Harbor task bundles
such as TaskTrove. Harbor owns container setup, the agent loop, and verifier
execution; Ultra supplies the selected worker model and records the result.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness

HARBOR_BIN_ENV = "ULTRA_HARBOR_BIN"
HARBOR_WORKDIR_ENV = "ULTRA_HARBOR_WORKDIR"


def _harbor_binary() -> str | None:
    configured = os.environ.get(HARBOR_BIN_ENV)
    if configured:
        return configured if shutil.which(configured) or Path(configured).exists() else None
    return shutil.which("harbor") or shutil.which("sb")


def _harbor_task_asset(task: TaskSpec) -> dict[str, Any] | None:
    for asset in task.input.assets:
        if isinstance(asset, dict) and isinstance(asset.get("harbor_task"), dict):
            return dict(asset["harbor_task"])
    expected = task.grader.expected_answer
    if isinstance(expected, dict) and isinstance(expected.get("harbor_task"), dict):
        return dict(expected["harbor_task"])
    return None


def _user_instruction(task: TaskSpec) -> str:
    for message in reversed(task.input.messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _augment_instruction(task: TaskSpec, step: StepInput) -> str:
    parts = [_user_instruction(task).strip()]
    if step.subtask.strip():
        parts.append(f"Subtask:\n{step.subtask.strip()}")
    if step.prior_artifacts:
        parts.append("Prior worker outputs:")
        for artifact in step.prior_artifacts:
            parts.append(
                "\n".join(
                    [
                        f"- step: {artifact.get('step_index')}",
                        f"  worker: {artifact.get('worker_name')}",
                        f"  harness: {artifact.get('harness')}",
                        f"  subtask: {artifact.get('subtask')}",
                        f"  result:\n{artifact.get('response', '')}",
                    ]
                )
            )
    return "\n\n".join(part for part in parts if part)


def _prepare_task_copy(task_dir: Path, run_root: Path, task: TaskSpec, step: StepInput) -> Path:
    task_copy = run_root / "tasks" / task_dir.name
    if task_copy.exists():
        shutil.rmtree(task_copy)
    shutil.copytree(task_dir, task_copy)
    (task_copy / "instruction.md").write_text(_augment_instruction(task, step))
    return task_copy


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def harbor_rewards(job_dir: Path) -> list[float]:
    """Extract verifier rewards from Harbor per-trial result files."""

    rewards: list[float] = []
    for result_path in sorted(job_dir.rglob("result.json")):
        data = _load_json(result_path)
        if not isinstance(data, dict):
            continue
        verifier = data.get("verifier_result")
        if not isinstance(verifier, dict):
            continue
        reward = None
        nested = verifier.get("rewards")
        if isinstance(nested, dict):
            reward = nested.get("reward")
        if reward is None:
            reward = verifier.get("reward")
        try:
            rewards.append(float(reward))
        except (TypeError, ValueError):
            continue
    return rewards


def harbor_grade_from_result(job_dir: Path, threshold: float) -> Grade:
    rewards = harbor_rewards(job_dir)
    if not rewards:
        return Grade(
            score=0.0,
            success=False,
            details={"error": "no Harbor verifier rewards found", "job_dir": str(job_dir)},
        )
    score = sum(rewards) / len(rewards)
    return Grade(
        score=score,
        success=score >= threshold,
        grader_ref=str(job_dir),
        details={"rewards": rewards, "job_dir": str(job_dir)},
    )


@register_harness
class TerminalSandboxHarborHarness:
    name = "terminal_sandbox"

    def __init__(self) -> None:
        self.last_job_dir: Path | None = None

    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult:
        harbor_bin = _harbor_binary()
        if harbor_bin is None:
            return StepResult(
                text="",
                error="Harbor CLI not found; install/activate Harbor before terminal_sandbox execution",
                termination="missing_harbor_cli",
            )

        asset = _harbor_task_asset(step.task)
        if asset is None:
            return StepResult(
                text="",
                error="terminal_sandbox task is missing a harbor_task asset",
                termination="missing_task_payload",
            )

        task_dir = Path(str(asset.get("task_dir", ""))).expanduser()
        if not task_dir.exists():
            return StepResult(
                text="",
                error=f"Harbor task_dir does not exist: {task_dir}",
                termination="missing_task_payload",
            )

        work_root = Path(os.environ.get(HARBOR_WORKDIR_ENV, ".ultra_harbor_runs")).resolve()
        run_root = work_root / f"{step.task.task_id}__s{step.step_index}__{int(time.time())}"
        jobs_dir = run_root / "jobs"
        run_root.mkdir(parents=True, exist_ok=True)
        task_copy = _prepare_task_copy(task_dir, run_root, step.task, step)

        model = pool.model_for(step.worker_id)
        agent = str(asset.get("agent") or "terminus-2")
        env_type = str(asset.get("environment") or "docker")
        job_name = f"{step.task.task_id}__s{step.step_index}"
        cmd = [
            harbor_bin,
            "jobs",
            "start",
            "--yes",
            "-p",
            str(task_copy),
            "--agent",
            agent,
            "--model",
            model,
            "--env",
            env_type,
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--job-name",
            job_name,
            "--jobs-dir",
            str(jobs_dir),
        ]

        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=step.task.environment.wall_time_seconds or 900,
            check=False,
        )
        self.last_job_dir = jobs_dir / job_name
        payload = {
            "job_dir": str(self.last_job_dir),
            "task_dir": str(task_copy),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        return StepResult(
            text=json.dumps(payload, sort_keys=True),
            error=None if proc.returncode == 0 else f"Harbor exited {proc.returncode}",
            termination="completed" if proc.returncode == 0 else "harbor_failed",
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        job_dir = self.last_job_dir
        if final.text:
            data = _load_json_from_text(final.text)
            if isinstance(data, dict) and data.get("job_dir"):
                job_dir = Path(str(data["job_dir"]))
        if job_dir is None:
            return Grade(score=0.0, success=False, details={"error": "no Harbor job_dir recorded"})
        return harbor_grade_from_result(job_dir, task.grader.success_threshold)


def _load_json_from_text(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
