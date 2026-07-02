import json
from pathlib import Path

import pytest

from ultra.config import WorkerSpec
from ultra.harness import HARNESS_REGISTRY, StepInput
from ultra.harness.harbor import (
    HarborVerifierUnavailable,
    TerminalSandboxHarborHarness,
    _agent_kwargs,
    _harbor_timeout_seconds,
    harbor_grade_from_result,
    harbor_rewards,
    _harbor_provider_model,
)
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


def test_harbor_grade_raises_on_environment_setup_failure(tmp_path):
    # A compose/environment setup failure means the verifier never ran: an infrastructure
    # failure, not a gradeable 0.5. harbor_grade_from_result must raise so the executor
    # excludes the rollout (grader_crash) instead of recording a fake valid-incorrect 0.5.
    job = tmp_path / "jobs" / "job"
    trial = job / "trial-1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "VerificationNotCompletedError",
                    "exception_message": "Docker compose command failed. failed to create network: all predefined address pools have been fully subnetted",
                },
                "verifier_result": None,
            }
        )
    )

    with pytest.raises(HarborVerifierUnavailable):
        harbor_grade_from_result(job, 1.0)


def test_harbor_provider_model_maps_to_active_slug(monkeypatch):
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")

    assert _harbor_provider_model("z-ai/glm-5.2") == "openai/z-ai/glm-5.2"

    monkeypatch.setenv("ULTRA_ALLOW_YUNWU", "1")
    assert _harbor_provider_model("gpt") == "openai/gpt-5.5"
    assert _harbor_provider_model("glm", "yunwu") == "openai/glm-5.2"


def test_harbor_provider_model_blocks_yunwu_without_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")
    monkeypatch.delenv("ULTRA_ALLOW_YUNWU", raising=False)

    with pytest.raises(RuntimeError, match="Yunwu Harbor call"):
        _harbor_provider_model("gpt")


def test_harbor_max_tokens_is_opt_in(monkeypatch):
    asset = {"agent_kwargs": {}}

    kwargs = _agent_kwargs(asset, "medium", "openrouter", Sampling(max_tokens=1024, reasoning_effort="high"))

    assert "llm_call_kwargs" not in kwargs
    assert kwargs["reasoning_effort"] == "high"

    monkeypatch.setenv("ULTRA_HARBOR_MAX_TOKENS", "4096")
    capped = _agent_kwargs(asset, "medium", "openrouter", Sampling(max_tokens=1024, reasoning_effort="high"))

    assert capped["llm_call_kwargs"] == {"max_tokens": 4096}


def test_harbor_timeout_override_is_opt_in(monkeypatch, tmp_path):
    task = _task(_bundle(tmp_path))

    assert _harbor_timeout_seconds("short", task) == 120

    monkeypatch.setenv("ULTRA_HARBOR_TIMEOUT_SECONDS", "30")
    assert _harbor_timeout_seconds("short", task) == 30

    monkeypatch.setenv("ULTRA_HARBOR_TIMEOUT_SECONDS", "not-a-number")
    assert _harbor_timeout_seconds("short", task) == 120


@pytest.mark.asyncio
async def test_harbor_harness_builds_job_and_grades_fake_result(monkeypatch, tmp_path):
    monkeypatch.setattr("ultra.harness.harbor._harbor_binary", lambda: "/fake/harbor")
    monkeypatch.setenv("ULTRA_HARBOR_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    def fake_run(cmd, **kwargs):
        assert "test-secret" not in " ".join(cmd)
        assert kwargs["env"]["OPENAI_API_KEY"] == "test-secret"
        assert cmd[cmd.index("--model") + 1] == "openai/z-ai/glm-5.2"
        assert "api_base=https://openrouter.ai/api/v1" in cmd
        assert "max_turns=8" in cmd
        assert not any(arg.startswith("llm_call_kwargs=") for arg in cmd)
        assert kwargs["timeout"] == 120

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
    assert payload["provider"] == {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "z-ai/glm-5.2",
        "harbor_model": "openai/z-ai/glm-5.2",
        "key_env": "OPENROUTER_API_KEY",
    }
    assert payload["telemetry_ref"] == result.tool_events_ref
    assert result.command_log_ref == result.tool_events_ref
    assert result.messages_ref and Path(result.messages_ref).name == "instruction.md"
    assert result.session_ref and "tasktrove_harbor__demo-task" in Path(result.session_ref).name
    telemetry = json.loads(Path(result.tool_events_ref).read_text())
    assert telemetry["provider"] == payload["provider"]
    assert telemetry["budget"]["max_turns"] == 8
    assert telemetry["status"] == "finished"
    assert "test-secret" not in Path(result.tool_events_ref).read_text()
    assert grade.success is True


@pytest.mark.asyncio
async def test_harbor_harness_returns_structured_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr("ultra.harness.harbor._harbor_binary", lambda: "/fake/harbor")
    monkeypatch.setenv("ULTRA_HARBOR_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ULTRA_HARBOR_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    def fake_timeout_run(cmd, **kwargs):
        raise __import__("subprocess").TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr("subprocess.run", fake_timeout_run)
    harness = TerminalSandboxHarborHarness()
    task = _task(_bundle(tmp_path))

    result = await harness.run_step(
        StepInput(task=task, subtask="Solve carefully", worker_id="harbor_glm_agent"),
        _pool(),
        Sampling(),
    )

    payload = json.loads(result.text)
    assert result.termination == "harbor_timeout"
    assert result.error == "Harbor timed out after 7s"
    assert payload["timeout_seconds"] == 7
    assert payload["job_dir"]
    assert payload["provider"]["name"] == "openrouter"
    assert payload["provider"]["model"] == "z-ai/glm-5.2"
    assert payload["telemetry_ref"] == result.tool_events_ref
    telemetry = json.loads(Path(result.tool_events_ref).read_text())
    assert telemetry["status"] == "timeout"
    assert telemetry["termination"] == "harbor_timeout"
    assert telemetry["budget"]["timeout_seconds"] == 7


@pytest.mark.asyncio
async def test_harbor_harness_isolates_parallel_rollout_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr("ultra.harness.harbor._harbor_binary", lambda: "/fake/harbor")
    monkeypatch.setenv("ULTRA_HARBOR_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    seen: list[tuple[Path, str]] = []

    def fake_run(cmd, **kwargs):
        jobs_dir = Path(cmd[cmd.index("--jobs-dir") + 1])
        job_name = cmd[cmd.index("--job-name") + 1]
        seen.append((jobs_dir, job_name))
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
    task = _task(_bundle(tmp_path))

    first = await TerminalSandboxHarborHarness().run_step(
        StepInput(
            task=task,
            subtask="Solve carefully",
            worker_id="harbor_glm_agent",
            rollout_id="job_00176",
        ),
        _pool(),
        Sampling(),
    )
    second = await TerminalSandboxHarborHarness().run_step(
        StepInput(
            task=task,
            subtask="Solve carefully",
            worker_id="harbor_glm_agent",
            rollout_id="job_00178",
        ),
        _pool(),
        Sampling(),
    )

    assert first.termination == "completed"
    assert second.termination == "completed"
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert seen[0][0] != seen[1][0]


@pytest.mark.asyncio
async def test_harbor_harness_clears_stale_rerun_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("ultra.harness.harbor._harbor_binary", lambda: "/fake/harbor")
    monkeypatch.setenv("ULTRA_HARBOR_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    calls: list[tuple[Path, str]] = []

    def fake_run(cmd, **kwargs):
        jobs_dir = Path(cmd[cmd.index("--jobs-dir") + 1])
        job_name = cmd[cmd.index("--job-name") + 1]
        calls.append((jobs_dir, job_name))
        assert not (jobs_dir / job_name / "stale.txt").exists()

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
    task = _task(_bundle(tmp_path))
    step = StepInput(
        task=task,
        subtask="Solve carefully",
        worker_id="harbor_glm_agent",
        rollout_id="job_repeat",
    )

    first = await TerminalSandboxHarborHarness().run_step(step, _pool(), Sampling())
    assert first.termination == "completed"
    jobs_dir, job_name = calls[0]
    stale = jobs_dir / job_name / "stale.txt"
    stale.write_text("old run")

    second = await TerminalSandboxHarborHarness().run_step(step, _pool(), Sampling())
    assert second.termination == "completed"
    assert len(calls) == 2
    assert calls[0] == calls[1]
