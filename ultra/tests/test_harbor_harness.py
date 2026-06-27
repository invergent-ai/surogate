import json
from pathlib import Path

import pytest

from ultra.config import WorkerSpec
from ultra.harness import HARNESS_REGISTRY, StepInput
from ultra.harness.harbor import TerminalSandboxHarborHarness, harbor_grade_from_result, harbor_rewards
from ultra.schemas import EnvironmentSpec, GraderSpec, SourceRef, SplittingSpec, TaskInput, TaskSpec
from ultra.workers import FakeProvider, Sampling, WorkerPool


def _bundle(root: Path) -> Path:
    task = root / "task-a"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Create /app/out.txt with ok.\n")
    (task / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "[metadata]",
                'task_id = "demo-task"',
                "[verifier]",
                "timeout_sec = 120",
                "[environment]",
                'docker_image = "example/harbor-task:latest"',
            ]
        )
    )
    (task / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    return task


def _task(task_dir: Path) -> TaskSpec:
    return TaskSpec(
        task_id="tasktrove_harbor__demo-task",
        capability="terminal_agentic",
        source=SourceRef(name="tasktrove_harbor", version="v3", policy="pool_only"),
        input=TaskInput(
            messages=[{"role": "user", "content": "Create /app/out.txt with ok."}],
            assets=[{"harbor_task": {"task_dir": str(task_dir), "agent": "terminus-2", "environment": "docker"}}],
        ),
        environment=EnvironmentSpec(harness="terminal_sandbox", wall_time_seconds=120),
        grader=GraderSpec(type="harbor_verifier"),
        splitting=SplittingSpec(group_id="g", split="pool_validation"),
    )


def _pool() -> WorkerPool:
    return WorkerPool([WorkerSpec(worker_id="harbor_glm_agent", model="z-ai/glm-5.2")], FakeProvider())


def test_terminal_sandbox_harness_registered():
    assert HARNESS_REGISTRY["terminal_sandbox"] is TerminalSandboxHarborHarness


@pytest.mark.asyncio
async def test_harbor_harness_fails_closed_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("ultra.harness.harbor._harbor_binary", lambda: None)
    harness = TerminalSandboxHarborHarness()

    result = await harness.run_step(
        StepInput(task=_task(_bundle(tmp_path)), subtask="Solve", worker_id="harbor_glm_agent"),
        _pool(),
        Sampling(),
    )

    assert result.termination == "missing_harbor_cli"
    assert "Harbor CLI not found" in result.error


def test_harbor_reward_parser_and_grade(tmp_path):
    job = tmp_path / "jobs" / "job"
    trial = job / "trial-1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
    )

    assert harbor_rewards(job) == [1.0]
    grade = harbor_grade_from_result(job, 1.0)
    assert grade.success is True
    assert grade.score == 1.0


@pytest.mark.asyncio
async def test_harbor_harness_builds_job_and_grades_fake_result(monkeypatch, tmp_path):
    monkeypatch.setattr("ultra.harness.harbor._harbor_binary", lambda: "/fake/harbor")
    monkeypatch.setenv("ULTRA_HARBOR_WORKDIR", str(tmp_path / "runs"))

    def fake_run(cmd, **kwargs):
        jobs_dir = Path(cmd[cmd.index("--jobs-dir") + 1])
        job_name = cmd[cmd.index("--job-name") + 1]
        trial = jobs_dir / job_name / "trial-1"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
        )

        class Proc:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
    harness = TerminalSandboxHarborHarness()
    task = _task(_bundle(tmp_path))

    result = await harness.run_step(
        StepInput(task=task, subtask="Solve carefully", worker_id="harbor_glm_agent"),
        _pool(),
        Sampling(),
    )
    grade = harness.grade(task, result)

    payload = json.loads(result.text)
    assert result.termination == "completed"
    assert payload["returncode"] == 0
    assert grade.success is True
