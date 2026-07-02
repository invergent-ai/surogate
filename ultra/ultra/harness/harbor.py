"""Harbor-backed terminal sandbox execution.

This harness lets Ultra route ``terminal_sandbox`` steps to Harbor task bundles
such as TaskTrove. Harbor owns container setup, the agent loop, and verifier
execution; Ultra supplies the selected worker model and records the result.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..providers import MODELS, assert_live_provider_allowed, provider as provider_config
from ..providers import routed_provider_name, routed_slug
from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness, wall_time_cap_seconds
from .repo_artifacts import artifact_ref, write_json

HARBOR_BIN_ENV = "ULTRA_HARBOR_BIN"
HARBOR_WORKDIR_ENV = "ULTRA_HARBOR_WORKDIR"
HARBOR_PROVIDER_ENV = "ULTRA_HARBOR_PROVIDER"
HARBOR_MAX_TOKENS_ENV = "ULTRA_HARBOR_MAX_TOKENS"
HARBOR_TIMEOUT_ENV = "ULTRA_HARBOR_TIMEOUT_SECONDS"

_MAX_TURNS_BY_BUDGET = {
    "short": 4,
    "medium": 8,
    "long": 20,
    "max": 40,
}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned[:120] or "run"


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


def _logical_name_for_model(model: str, provider_name: str) -> str | None:
    if model in MODELS:
        return model
    for logical, mapping in MODELS.items():
        if model in mapping.values() or model == f"openai/{mapping.get(provider_name, '')}":
            return logical
    return None


def _harbor_provider_model(model: str, provider_name: str | None = None) -> str:
    """Return the LiteLLM/OpenAI-compatible model name Harbor should call."""

    override = provider_name or os.environ.get(HARBOR_PROVIDER_ENV)
    selected = routed_provider_name(model, override)
    assert_live_provider_allowed(selected, model=model, context="Harbor call")
    logical = _logical_name_for_model(model, selected)
    resolved = routed_slug(logical, selected) if logical else model
    if resolved.startswith("openai/"):
        return resolved
    return f"openai/{resolved}"


def _format_agent_kwarg(key: str, value: Any) -> str:
    if isinstance(value, (dict, list, bool, int, float)) or value is None:
        encoded = json.dumps(value)
    else:
        encoded = str(value)
    return f"{key}={encoded}"


def _agent_kwargs(
    asset: dict[str, Any],
    budget: str,
    provider_name: str,
    sampling: Sampling,
) -> dict[str, Any]:
    raw = asset.get("agent_kwargs")
    kwargs = dict(raw) if isinstance(raw, dict) else {}
    kwargs.setdefault("api_base", provider_config(provider_name).get("base_url"))
    kwargs.setdefault("max_turns", _MAX_TURNS_BY_BUDGET.get(budget, _MAX_TURNS_BY_BUDGET["medium"]))
    llm_call_kwargs = dict(kwargs.get("llm_call_kwargs") or {})
    max_tokens = os.environ.get(HARBOR_MAX_TOKENS_ENV)
    if max_tokens and "max_tokens" not in llm_call_kwargs:
        llm_call_kwargs["max_tokens"] = int(max_tokens)
    if llm_call_kwargs:
        kwargs["llm_call_kwargs"] = llm_call_kwargs
    if sampling.reasoning_effort is not None:
        kwargs.setdefault("reasoning_effort", sampling.reasoning_effort)
    return {key: value for key, value in kwargs.items() if value is not None}


def _harbor_timeout_seconds(budget: str, task: TaskSpec) -> int | None:
    timeout = wall_time_cap_seconds(
        budget,
        task_cap=task.environment.wall_time_seconds,
        harness_cap=900,
    )
    override = os.environ.get(HARBOR_TIMEOUT_ENV)
    if override:
        try:
            override_timeout = int(float(override))
        except ValueError:
            override_timeout = None
        if override_timeout is not None and override_timeout > 0:
            timeout = min(timeout, override_timeout) if timeout is not None else override_timeout
    return timeout


def _harbor_env(provider_name: str) -> dict[str, str] | None:
    selected = provider_name
    key_env = provider_config(selected).get("key_env")
    if not key_env:
        return os.environ.copy()
    key = os.environ.get(str(key_env))
    if not key:
        return None
    env = os.environ.copy()
    # Harbor/Terminus uses LiteLLM against an OpenAI-compatible base URL. Keep the
    # provider key out of CLI argv and persisted Harbor config.
    env["OPENAI_API_KEY"] = key
    return env


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


def _harbor_failure_details(job_dir: Path) -> dict[str, Any]:
    exception_types: list[str] = []
    messages: list[str] = []
    for result_path in sorted(job_dir.rglob("result.json")):
        data = _load_json(result_path)
        if not isinstance(data, dict):
            continue
        exception = data.get("exception_info")
        if isinstance(exception, dict):
            exception_type = str(exception.get("exception_type") or "")
            exception_message = str(exception.get("exception_message") or "")
            if exception_type:
                exception_types.append(exception_type)
            if exception_message:
                messages.append(exception_message)
        stats = data.get("stats")
        if isinstance(stats, dict):
            evals = stats.get("evals")
            if isinstance(evals, dict):
                for eval_result in evals.values():
                    if not isinstance(eval_result, dict):
                        continue
                    exception_stats = eval_result.get("exception_stats")
                    if isinstance(exception_stats, dict):
                        exception_types.extend(str(key) for key in exception_stats)

    details: dict[str, Any] = {
        "error": "no Harbor verifier rewards found",
        "job_dir": str(job_dir),
    }
    if exception_types:
        details["harbor_exception_types"] = sorted(set(exception_types))
    if messages:
        message = messages[0]
        details["harbor_exception_message"] = message[:2000]
        lowered = message.lower()
        if (
            "docker compose command failed" in lowered
            or "failed to create network" in lowered
            or "all predefined address pools" in lowered
        ):
            details["error"] = "harbor environment setup failed"
    return details


def _harbor_agent_paths(job_dir: Path) -> dict[str, str | None]:
    trajectory = next(iter(sorted(job_dir.rglob("agent/trajectory.json"))), None)
    recording = next(iter(sorted(job_dir.rglob("agent/recording.cast"))), None)
    pane = next(iter(sorted(job_dir.rglob("agent/terminus_2.pane"))), None)
    result = job_dir / "result.json"
    config = job_dir / "config.json"
    return {
        "job_result_ref": artifact_ref(result) if result.exists() else None,
        "job_config_ref": artifact_ref(config) if config.exists() else None,
        "trajectory_ref": artifact_ref(trajectory) if trajectory and trajectory.exists() else None,
        "recording_ref": artifact_ref(recording) if recording and recording.exists() else None,
        "pane_ref": artifact_ref(pane) if pane and pane.exists() else None,
    }


def _harbor_job_stats(job_dir: Path) -> dict[str, Any]:
    result = _load_json(job_dir / "result.json")
    stats = result.get("stats") if isinstance(result, dict) else None
    if not isinstance(stats, dict):
        return {}
    return {
        "n_input_tokens": stats.get("n_input_tokens"),
        "n_cache_tokens": stats.get("n_cache_tokens"),
        "n_output_tokens": stats.get("n_output_tokens"),
        "cost_usd": stats.get("cost_usd"),
        "n_completed_trials": stats.get("n_completed_trials"),
        "n_running_trials": stats.get("n_running_trials"),
        "n_errored_trials": stats.get("n_errored_trials"),
    }


def _safe_llm_call_kwargs(agent_kwargs: dict[str, Any]) -> dict[str, Any]:
    raw = agent_kwargs.get("llm_call_kwargs")
    if not isinstance(raw, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, value in raw.items():
        lowered = str(key).lower()
        if any(secret in lowered for secret in ("key", "token", "secret", "auth", "password")):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
    return safe


def _write_harbor_telemetry(path: Path, payload: dict[str, Any]) -> str:
    return write_json(path, payload)


class HarborVerifierUnavailable(RuntimeError):
    """The Harbor verifier produced no reward (e.g. ``docker compose up`` / environment
    setup failed, so the verifier never ran). The rollout cannot be graded, so this is
    raised: the executor records it as a grader/harness failure (reward null, excluded
    from training) instead of a fake valid-but-incorrect 0.5."""


def harbor_grade_from_result(job_dir: Path, threshold: float) -> Grade:
    rewards = harbor_rewards(job_dir)
    if not rewards:
        details = _harbor_failure_details(job_dir)
        raise HarborVerifierUnavailable(str(details.get("error") or "no Harbor verifier rewards found"))
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
        run_id = _safe_name(step.rollout_id or f"{os.getpid()}-{time.time_ns()}")
        worker_name = _safe_name(str(step.worker_id))
        task_name = _safe_name(step.task.task_id)
        run_root = work_root / f"{task_name}__{run_id}__s{step.step_index}__w{worker_name}"
        jobs_dir = run_root / "jobs"
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        task_copy = _prepare_task_copy(task_dir, run_root, step.task, step)

        requested_model = pool.model_for(step.worker_id)
        provider_name = routed_provider_name(requested_model, os.environ.get(HARBOR_PROVIDER_ENV))
        assert_live_provider_allowed(
            provider_name,
            model=requested_model,
            context="Harbor call",
        )
        model = _harbor_provider_model(requested_model, provider_name)
        agent = str(asset.get("agent") or "terminus-2")
        env_type = str(asset.get("environment") or "docker")
        job_name = f"{task_name}__{run_id}__s{step.step_index}"
        agent_kwargs = _agent_kwargs(asset, step.budget, provider_name, sampling)
        provider_cfg = provider_config(provider_name)
        provider_model = model.removeprefix("openai/")
        telemetry_path = run_root / "ultra_harbor_telemetry.json"
        telemetry: dict[str, Any] = {
            "task_id": step.task.task_id,
            "rollout_id": step.rollout_id,
            "step_index": step.step_index,
            "worker_id": str(step.worker_id),
            "requested_model": requested_model,
            "provider": {
                "name": provider_name,
                "base_url": provider_cfg.get("base_url"),
                "model": provider_model,
                "harbor_model": model,
                "key_env": provider_cfg.get("key_env"),
            },
            "harbor": {
                "agent": agent,
                "environment": env_type,
                "job_name": job_name,
                "jobs_dir": str(jobs_dir),
                "task_dir": str(task_copy),
            },
            "budget": {
                "step_budget": step.budget,
                "timeout_seconds": _harbor_timeout_seconds(step.budget, step.task),
                "max_turns": agent_kwargs.get("max_turns"),
                "reasoning_effort": agent_kwargs.get("reasoning_effort"),
                "llm_call_kwargs": _safe_llm_call_kwargs(agent_kwargs),
            },
            "status": "starting",
        }
        telemetry_ref = _write_harbor_telemetry(telemetry_path, telemetry)
        env = _harbor_env(provider_name)
        if env is None:
            key_env = provider_cfg.get("key_env")
            return StepResult(
                text="",
                error=f"terminal_sandbox Harbor provider key is missing: {key_env}",
                termination="missing_provider_key",
                tool_events_ref=telemetry_ref,
                command_log_ref=telemetry_ref,
                artifact_dir=str(run_root),
            )
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
        for key, value in agent_kwargs.items():
            cmd.extend(["--agent-kwarg", _format_agent_kwarg(key, value)])

        self.last_job_dir = jobs_dir / job_name
        timeout = _harbor_timeout_seconds(step.budget, step.task)
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            telemetry.update(
                {
                    "status": "timeout",
                    "termination": "harbor_timeout",
                    "harbor": telemetry["harbor"] | _harbor_agent_paths(self.last_job_dir),
                    "stats": _harbor_job_stats(self.last_job_dir),
                }
            )
            telemetry_ref = _write_harbor_telemetry(telemetry_path, telemetry)
            payload = {
                "job_dir": str(self.last_job_dir),
                "task_dir": str(task_copy),
                "timeout_seconds": timeout,
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                "provider": telemetry["provider"],
                "harbor": telemetry["harbor"],
                "telemetry_ref": telemetry_ref,
            }
            return StepResult(
                text=json.dumps(payload, sort_keys=True),
                error=f"Harbor timed out after {timeout}s",
                termination="harbor_timeout",
                session_ref=str(self.last_job_dir),
                messages_ref=str(task_copy / "instruction.md"),
                tool_events_ref=telemetry_ref,
                command_log_ref=telemetry_ref,
                artifact_dir=str(run_root),
            )
        termination = "completed" if proc.returncode == 0 else "harbor_failed"
        telemetry.update(
            {
                "status": "finished",
                "termination": termination,
                "returncode": proc.returncode,
                "harbor": telemetry["harbor"] | _harbor_agent_paths(self.last_job_dir),
                "stats": _harbor_job_stats(self.last_job_dir),
            }
        )
        telemetry_ref = _write_harbor_telemetry(telemetry_path, telemetry)
        payload = {
            "job_dir": str(self.last_job_dir),
            "task_dir": str(task_copy),
            "returncode": proc.returncode,
            "timeout_seconds": timeout,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "provider": telemetry["provider"],
            "harbor": telemetry["harbor"],
            "telemetry_ref": telemetry_ref,
        }
        return StepResult(
            text=json.dumps(payload, sort_keys=True),
            error=None if proc.returncode == 0 else f"Harbor exited {proc.returncode}",
            termination=termination,
            session_ref=str(self.last_job_dir),
            messages_ref=str(task_copy / "instruction.md"),
            tool_events_ref=telemetry_ref,
            command_log_ref=telemetry_ref,
            artifact_dir=str(run_root),
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        job_dir = self.last_job_dir
        if final.text:
            data = _load_json_from_text(final.text)
            if isinstance(data, dict) and data.get("job_dir"):
                job_dir = Path(str(data["job_dir"]))
        if job_dir is None:
            raise HarborVerifierUnavailable("no Harbor job_dir recorded")
        return harbor_grade_from_result(job_dir, task.grader.success_threshold)

    def close(self) -> None:
        """Tear down this rollout's Harbor compose project (containers + network).

        Harbor leaves each rollout's compose project running, and every project holds a
        Docker network. Without teardown the ~31-network address pool is exhausted after
        a few dozen rollouts, after which `docker compose up` fails and the verifier
        never runs ("no Harbor verifier rewards found"). The compose project name is the
        trial directory's basename (Docker-lowercased); we remove only this rollout's
        resources (matched by the compose project label), so it is safe under
        concurrency.
        """
        job_dir = self.last_job_dir
        if job_dir is None or not job_dir.exists():
            return
        try:
            projects = {t.name.lower() for t in job_dir.iterdir() if t.is_dir()}
        except OSError:
            return
        for project in projects:
            flt = f"label=com.docker.compose.project={project}"
            try:
                cids = subprocess.run(
                    ["docker", "ps", "-aq", "--filter", flt],
                    capture_output=True, text=True, timeout=30, check=False,
                ).stdout.split()
                if cids:
                    subprocess.run(["docker", "rm", "-f", *cids], capture_output=True, timeout=120, check=False)
                nids = subprocess.run(
                    ["docker", "network", "ls", "-q", "--filter", flt],
                    capture_output=True, text=True, timeout=30, check=False,
                ).stdout.split()
                if nids:
                    subprocess.run(["docker", "network", "rm", *nids], capture_output=True, timeout=60, check=False)
            except Exception:  # noqa: BLE001 - cleanup is best-effort, never fail the rollout
                pass


def _load_json_from_text(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
