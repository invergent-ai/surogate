"""opencode_repo execution — agentic coding via OpenCode in SWE-smith containers.

Reuses the proven container mechanics from ``director/per_step_coding.py`` (start + branch
checkout, ``oc run`` inside the ``testbed`` conda env, git-diff capture) and director's
SWE-smith grader. The worker is driven by OpenCode INSIDE the container; ultra's pool only
supplies the worker's OpenRouter model slug.

Standalone-core preserved: the ``director``/``swebench`` imports are LAZY (inside the run
function), so ultra's own venv can import this module. To actually RUN it you need the
agentic env (director/.venv: docker + swebench + swesmith + mini-swe-agent + the ``oc``
binary + SWE-smith images), with ultra on PYTHONPATH.

Workspace lineage (ultra-data §6): one container = one workflow node's workspace snapshot.
  access=[]        -> fresh container from the base image, checked out to the buggy branch (Rule A)
  access=[j]       -> CONTINUE in step j's container (inherits its edits) (Rule B)
  access=[j,k,...] -> fresh container; predecessors' patches handed in as text artifacts (Rule C)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

from ..providers import assert_live_provider_allowed
from ..providers import provider as _provider_cfg
from ..providers import routed_provider_name, routed_slug
from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness, wall_time_cap_seconds
from .repo_artifacts import artifact_ref, write_json, write_repo_state, write_text

OC_BIN = os.environ.get("ULTRA_OC_BIN", os.path.expanduser("~/.opencode/bin/opencode"))
CONDA_ACTIVATE = os.environ.get(
    "ULTRA_CONDA_ACTIVATE",
    "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed",
)
TESTBED = os.environ.get("ULTRA_TESTBED", "/testbed")
OC_TIMEOUT = int(os.environ.get("ULTRA_OC_TIMEOUT", "600"))

# Default key env retained for tests/backward imports. Runtime routing is per worker.
KEY_ENV = "OPENROUTER_API_KEY"

_SCAFFOLD_WORKER_LOGICAL = {
    "opencode_kimi_builder": "kimi",
    "opencode_mimo_repair": "mimo",
    "opencode_glm_builder": "glm",
    "opencode_flash": "flash",
    "kimi-code": "kimi",
    "kimi": "kimi",
    "mimo": "mimo",
    "glm": "glm",
    "flash": "flash",
    "minimax": "minimax",
    "deepseek-pro": "deepseek-pro",
    "opus": "opus",
    "gpt": "gpt",
    "gemini": "gemini",
}


def _sh(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(a, capture_output=True, text=True)


def _opencode_target(worker_id: str, model: str | None = None) -> str:
    return _SCAFFOLD_WORKER_LOGICAL.get(worker_id) or model or worker_id


def _opencode_route(worker_id: str, model: str | None = None) -> dict[str, str]:
    target = _opencode_target(worker_id, model)
    provider_name = routed_provider_name(target, os.environ.get("ULTRA_OC_PROVIDER"))
    assert_live_provider_allowed(provider_name, model=target, context="OpenCode call")
    cfg = _provider_cfg(provider_name)
    return {
        "provider_name": provider_name,
        "oc_provider": str(cfg["oc_provider"]),
        "key_env": os.environ.get("ULTRA_OC_KEY_ENV", str(cfg.get("key_env") or "")),
        "oc_config": os.environ.get("ULTRA_OC_CONFIG", str(cfg.get("oc_config") or "")),
        "slug": routed_slug(target, provider_name),
    }


def _opencode_slug(worker_id: str) -> str:
    return _opencode_route(worker_id)["slug"]


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _read_initial_patch(instance: dict[str, Any]) -> tuple[str | None, str | None]:
    ref = str(instance.get("initial_patch_ref") or "").strip()
    if not ref:
        return None, None
    path = Path(ref)
    if not path.exists() or not path.is_file():
        return None, f"initial_patch_ref not found: {ref}"
    try:
        text = path.read_text()
    except OSError as exc:
        return None, f"initial_patch_ref read failed: {type(exc).__name__}: {exc}"
    if not text.strip():
        return None, f"initial_patch_ref is empty: {ref}"
    return text, None


class OpenCodeContainer:
    """One container == one workflow node's mutable workspace (reused from per_step_coding.py)."""

    def __init__(
        self,
        image: str,
        instance_id: str,
        *,
        testbed: str = TESTBED,
        tests_dir: str | None = None,
        activate: str | None = CONDA_ACTIVATE,
        oc_config: str = "",
    ):
        self.image = image
        self.instance_id = instance_id
        self.testbed = testbed
        self.tests_dir = tests_dir
        self.activate = activate or ""
        self.oc_config = oc_config
        self.cid = ""

    def start(self) -> bool:
        """Fresh container from the base image, checked out to the buggy branch (the bug is
        per-branch; base is clean — see memory swesmith-harness-gotchas)."""
        run_args = ["docker", "run", "-d", "--rm", "-v", f"{OC_BIN}:/usr/local/bin/oc:ro"]
        if self.tests_dir:
            run_args += ["-v", f"{self.tests_dir}:/tests:ro"]
        if self.oc_config:  # custom provider (yunwu): mount its OpenCode config read-only
            run_args += ["-v", f"{self.oc_config}:/root/opencode.json:ro"]
        run_args += [self.image, "sleep", "9000"]
        cid = _sh(*run_args).stdout.strip()
        if not cid:
            return False
        if not self.instance_id:
            self.cid = cid
            return True
        co = _sh("docker", "exec", cid, "bash", "-c", f"cd {self.testbed} && git checkout {self.instance_id} 2>&1")
        blob = (co.stderr + co.stdout).lower()
        if "error" in blob and "set up to track" not in blob:
            _sh("docker", "rm", "-f", cid)
            return False
        self.cid = cid
        return True

    def run_worker(
        self,
        slug: str,
        prompt: str,
        *,
        key_env: str,
        oc_provider: str,
        oc_config: str = "",
        timeout: int | None = None,
    ) -> dict:
        """Drive OpenCode with the worker model on the current workspace.

        Returns ``{"status": "ok"|"timeout", "cost": float}``.
        ``cost`` is provider-reported telemetry when OpenCode emits it. Some
        providers, including Yunwu, may emit no cost; external spend tracking is
        authoritative in that case.
        Runs inside the testbed conda login shell so the agent's python/pytest resolve."""
        activate = f"{self.activate} && " if self.activate else ""
        inner = (
            f"{activate}exec /usr/local/bin/oc run --format json "
            f'-m {oc_provider}/{slug} --dangerously-skip-permissions "$1"'
        )
        # Pass only the env var name. Docker inherits the value from this process
        # without putting the secret in the local process argv.
        envs = ["-e", key_env, "-e", "HOME=/root"]
        if oc_config:  # custom provider config mounted at /root/opencode.json
            envs += ["-e", "OPENCODE_CONFIG=/root/opencode.json"]
        try:
            proc = subprocess.run(
                ["docker", "exec", *envs, "-w", self.testbed, self.cid, "bash", "-lc", inner, "_", prompt],
                capture_output=True, text=True, timeout=timeout or OC_TIMEOUT,
            )
            return {
                "status": "ok" if proc.returncode == 0 else "nonzero",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "cost": _opencode_cost(proc.stdout),
            }
        except subprocess.TimeoutExpired as exc:
            # Timing out ``docker exec`` does not stop the process in the container. A
            # surviving agent can otherwise keep editing while the next workflow step runs.
            _sh("docker", "exec", self.cid, "pkill", "-KILL", "-x", "oc")
            return {
                "status": "timeout",
                "returncode": None,
                "stdout": _text(exc.stdout),
                "stderr": _text(exc.stderr),
                "cost": 0.0,
            }

    def diff(self) -> str:
        _sh("docker", "exec", self.cid, "bash", "-c", f"cd {self.testbed} && git add -A")
        diff = _sh("docker", "exec", self.cid, "bash", "-c", f"cd {self.testbed} && git diff --cached").stdout
        return _strip_ignored_diff_entries(diff)

    def export_workspace(self, destination: Path) -> tuple[bool, str]:
        destination.mkdir(parents=True, exist_ok=True)
        proc = _sh("docker", "cp", f"{self.cid}:{self.testbed}/.", str(destination))
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()
        return True, ""

    def apply_initial_patch(self, patch: str) -> tuple[bool, str]:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.cid,
                "bash",
                "-lc",
                f"cd {self.testbed} && git apply --whitespace=nowarn -",
            ],
            input=patch,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()
        return True, ""

    def grade_deep_swe(self, diff: str, artifact_dir: Path | None = None) -> float:
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.cid,
                "bash",
                "-lc",
                "mkdir -p /logs/artifacts /logs/verifier && cat > /logs/artifacts/model.patch",
            ],
            input=diff,
            capture_output=True,
            text=True,
            check=False,
        )
        proc = subprocess.run(
            ["docker", "exec", self.cid, "bash", "-lc", "bash /tests/test.sh"],
            capture_output=True,
            text=True,
            timeout=OC_TIMEOUT,
            check=False,
        )
        reward = _sh(
            "docker",
            "exec",
            self.cid,
            "bash",
            "-lc",
            "cat /logs/verifier/reward.json 2>/dev/null || cat /logs/verifier/reward.txt 2>/dev/null || true",
        ).stdout
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            write_text(
                artifact_dir / "test_command.log",
                f"returncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}",
            )
            write_text(artifact_dir / "reward_raw.txt", reward)
        return _deep_swe_reward_from_text(reward)

    def close(self) -> None:
        if self.cid:
            subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)


_HIDDEN = ("IMPORTANT: the failing test that reproduces this bug is NOT present in the repository — "
           "the existing tests already pass and will keep passing, so running them does NOT prove the "
           "bug is fixed. Read the problem description, locate the faulty code, and correct the "
           "described behavior directly.")
_FIX = ("The repository in {testbed} has the bug described above. Fix it by editing the SOURCE "
        "(do NOT modify tests). " + _HIDDEN)
_CONT = ("The repository in {testbed} contains a previous engineer's partial attempt at fixing the bug "
         "above; it may be incomplete or wrong. Review the current code against the problem "
         "description, find what is still wrong, and correct the described behavior. Do NOT modify "
         "tests. " + _HIDDEN)
_SYNTH = ("Below are {n} candidate patches from independent attempts at the bug above; none is "
          "necessarily complete or correct. Produce ONE correct fix in {testbed} (edit the SOURCE, "
          "not tests), combining their strongest ideas. " + _HIDDEN)


def _step_prompt(
    problem: str,
    subtask: str,
    prior_diffs: list[str],
    continuing: bool,
    testbed: str = TESTBED,
) -> str:
    parts = [problem.strip(), ""]
    if prior_diffs:  # Rule C — synthesize independent candidate patches in a fresh workspace
        parts.append(_SYNTH.format(n=len(prior_diffs), testbed=testbed))
        for j, d in enumerate(prior_diffs):
            parts.append(f"\n--- candidate patch {j} ---\n{d}")
    elif continuing:  # Rule B — continue in the predecessor's workspace (edits already present)
        parts.append(_CONT.format(testbed=testbed))
    else:  # Rule A — fresh
        parts.append(_FIX.format(testbed=testbed))
    if subtask.strip():
        parts.append(f"\nYour subtask: {subtask}")
    return "\n".join(parts)


def _opencode_cost(stdout: str) -> float:
    cost = 0.0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        part = event.get("part") if isinstance(event, dict) else None
        if isinstance(part, dict):
            cost += float(part.get("cost", 0.0) or 0.0)
    return cost


def _deep_swe_reward_from_text(text: str) -> float:
    blob = (text or "").strip()
    if not blob:
        return 0.0
    try:
        data = json.loads(blob)
    except ValueError:
        try:
            return 1.0 if float(blob) >= 1.0 else 0.0
        except ValueError:
            return 0.0
    if isinstance(data, dict):
        return 1.0 if float(data.get("reward", 0.0) or 0.0) >= 1.0 else 0.0
    if isinstance(data, (int, float)):
        return 1.0 if float(data) >= 1.0 else 0.0
    return 0.0


def _strip_ignored_diff_entries(diff: str) -> str:
    """Drop generated bytecode/cache files from source patches."""

    kept: list[str] = []
    block: list[str] = []

    def ignored(header: str) -> bool:
        return "__pycache__/" in header or ".pyc " in header or header.rstrip().endswith(".pyc")

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if block and not ignored(block[0]):
                kept.extend(block)
            block = [line]
        elif block:
            block.append(line)
        else:
            kept.append(line)
    if block and not ignored(block[0]):
        kept.extend(block)
    return "".join(kept)


def _user_problem(task: TaskSpec) -> str:
    for message in reversed(task.input.messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _instance_from_task(task: TaskSpec) -> dict | None:
    for asset in task.input.assets:
        if isinstance(asset, dict):
            if isinstance(asset.get("opencode_instance"), dict):
                return dict(asset["opencode_instance"])
            if "image_name" in asset:
                return dict(asset)
    expected = task.grader.expected_answer
    if isinstance(expected, dict):
        inst = expected.get("opencode_instance")
        if isinstance(inst, dict):
            return dict(inst)
    return None


def _normalize_instance(task: TaskSpec) -> dict | None:
    instance = _instance_from_task(task)
    if not instance:
        return None
    if not instance.get("image_name"):
        return None
    instance = dict(instance)
    instance.setdefault("instance_id", "")
    instance.setdefault("problem_statement", _user_problem(task))
    if not instance.get("problem_statement"):
        return None
    return instance


@register_harness
class OpenCodeRepoHarness:
    """Per-step OpenCode repository harness with Ultra workflow lineage semantics.

    The task must carry an ``opencode_instance`` asset compatible with the existing
    ``OpenCodeContainer`` runner. Missing payloads or missing provider keys return a
    normal failed ``StepResult`` instead of trying to infer hidden execution state.
    """

    name = "opencode"

    def __init__(self) -> None:
        self.containers: dict[int, OpenCodeContainer] = {}
        self.diffs: dict[int, str] = {}
        self.owned: list[OpenCodeContainer] = []
        self.instance: dict | None = None
        self.final_container: OpenCodeContainer | None = None
        self.total_cost = 0.0
        self.step_artifact_dirs: dict[int, Path] = {}

    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult:
        del sampling  # OpenCode owns the agent loop once the model slug is selected.

        route = _opencode_route(step.worker_id, pool.model_for(step.worker_id))
        key_env = route["key_env"]
        if key_env and not os.environ.get(key_env):
            return StepResult(
                text="",
                error=f"{key_env} is not set",
                termination="missing_provider_key",
            )

        instance = _normalize_instance(step.task)
        if instance is None:
            return StepResult(
                text="",
                error="opencode_repo task is missing an opencode_instance payload",
                termination="missing_task_payload",
            )
        self.instance = instance

        access = list(step.access)
        if len(access) == 1 and access[0] in self.containers:
            container = self.containers[access[0]]
            prior_diffs: list[str] = []
            continuing = True
        else:
            container = OpenCodeContainer(
                str(instance["image_name"]),
                str(instance.get("instance_id", "")),
                testbed=str(instance.get("testbed") or TESTBED),
                tests_dir=str(instance["tests_dir"]) if instance.get("tests_dir") else None,
                activate=str(instance.get("activate")) if "activate" in instance else CONDA_ACTIVATE,
                oc_config=route["oc_config"],
            )
            if not await asyncio.to_thread(container.start):
                return StepResult(
                    text="",
                    error="container start/checkout failed",
                    termination="container_start_failed",
                )
            self.owned.append(container)
            initial_patch, initial_patch_error = _read_initial_patch(instance)
            if initial_patch_error:
                return StepResult(
                    text="",
                    error=initial_patch_error,
                    termination="initial_patch_failed",
                )
            if initial_patch:
                applied, apply_error = await asyncio.to_thread(container.apply_initial_patch, initial_patch)
                if not applied:
                    return StepResult(
                        text="",
                        error=apply_error,
                        termination="initial_patch_failed",
                    )
            artifacts = {a.get("step_index"): a for a in step.prior_artifacts}
            prior_diffs = []
            for j in access:
                if j in self.diffs:
                    prior_diffs.append(self.diffs[j])
                elif j in artifacts:
                    prior_diffs.append(str(artifacts[j].get("response", "")))
            continuing = bool(prior_diffs or instance.get("initial_patch_ref"))

        prompt = _step_prompt(
            str(instance["problem_statement"]),
            step.subtask,
            prior_diffs,
            continuing,
            testbed=container.testbed,
        )
        timeout = wall_time_cap_seconds(
            step.budget,
            task_cap=step.task.environment.wall_time_seconds,
            harness_cap=OC_TIMEOUT,
        )
        run = await asyncio.to_thread(
            container.run_worker,
            route["slug"],
            prompt,
            key_env=key_env,
            oc_provider=route["oc_provider"],
            oc_config=route["oc_config"],
            timeout=timeout,
        )
        diff = await asyncio.to_thread(container.diff)
        cost = float(run.get("cost", 0.0) or 0.0)
        self.total_cost += cost
        self.diffs[step.step_index] = diff
        self.containers[step.step_index] = container
        self.final_container = container
        refs: dict[str, str | None] = {
            "messages_ref": None,
            "patch_ref": None,
            "tool_events_ref": None,
            "workspace_snapshot_ref": None,
        }
        step_artifact_dir = Path(step.artifact_dir) if step.artifact_dir else None
        if step_artifact_dir is not None:
            self.step_artifact_dirs[step.step_index] = step_artifact_dir
            refs["messages_ref"] = write_text(step_artifact_dir / "prompt.txt", prompt)
            if instance.get("initial_patch_ref"):
                write_json(
                    step_artifact_dir / "initial_patch.json",
                    {"initial_patch_ref": str(instance.get("initial_patch_ref"))},
                )
            refs["patch_ref"] = write_text(step_artifact_dir / "patch.diff", diff)
            refs["tool_events_ref"] = write_json(
                step_artifact_dir / "command.json",
                {
                    "harness": self.name,
                    "worker_id": step.worker_id,
                    "status": run.get("status"),
                    "returncode": run.get("returncode"),
                    "stdout": run.get("stdout", ""),
                    "stderr": run.get("stderr", ""),
                    "cost": cost,
                    "timeout_seconds": timeout,
                },
            )
            write_repo_state(step_artifact_dir / "repo_state.json", step.task, instance)
            workspace_dir = step_artifact_dir / "workspace_snapshot"
            exported, export_error = await asyncio.to_thread(container.export_workspace, workspace_dir)
            if exported:
                refs["workspace_snapshot_ref"] = artifact_ref(workspace_dir)
            else:
                write_json(step_artifact_dir / "workspace_export_error.json", {"error": export_error})

        status = str(run.get("status", "ok"))
        return StepResult(
            text=diff,
            cost_usd=cost,
            error=None if status == "ok" else status,
            termination="completed" if status == "ok" else status,
            session_ref=refs["workspace_snapshot_ref"],
            workspace_snapshot_ref=refs["workspace_snapshot_ref"],
            patch_ref=refs["patch_ref"],
            messages_ref=refs["messages_ref"],
            tool_events_ref=refs["tool_events_ref"],
            command_log_ref=refs["tool_events_ref"],
            artifact_dir=str(step_artifact_dir) if step_artifact_dir is not None else None,
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        try:
            if final.error and not final.text.strip():
                return Grade(score=0.0, success=False, details={"error": final.error})
            patch = _strip_ignored_diff_entries(final.text)
            if not patch.strip():
                return Grade(score=0.0, success=False, details={"error": "empty patch"})
            instance = self.instance or _normalize_instance(task)
            if not instance:
                return Grade(
                    score=0.0,
                    success=False,
                    details={"error": "opencode_repo task is missing an opencode_instance payload"},
                )
            if task.grader.type != "hidden_tests" and task.grader.type != "swesmith_hidden_tests":
                if task.grader.type == "swebench_verified_hidden_tests":
                    from ..acrouter_swebench import grade_swebench_verified_patch

                    instance_id = str(instance.get("swebench_instance_id") or instance.get("instance_id") or "")
                    if not instance_id:
                        return Grade(
                            score=0.0,
                            success=False,
                            details={"error": "missing swebench_instance_id"},
                        )
                    grade_dir = (
                        Path(final.artifact_dir) / "grade"
                        if final.artifact_dir
                        else Path(".ultra_swebench_grades") / task.task_id
                    )
                    result = grade_swebench_verified_patch(
                        instance_id=instance_id,
                        patch=patch,
                        image=str(instance["image_name"]),
                        log_dir=grade_dir,
                        eval_timeout=task.environment.wall_time_seconds or OC_TIMEOUT,
                        network="none",
                    )
                    reward = 1.0 if result.get("resolved") else 0.0
                    details = {"step_error": final.error} if final.error else {}
                    details.update(
                        {
                            "swebench_instance_id": instance_id,
                            "apply_ok": result.get("apply_ok"),
                            "resolved": result.get("resolved"),
                            "error": result.get("error"),
                            "redacted_log_path": result.get("redacted_log_path"),
                            "raw_log_retained": result.get("raw_log_retained"),
                        }
                    )
                    return Grade(score=reward, success=reward >= task.grader.success_threshold, details=details)
                if task.grader.type == "deep_swe_hidden_tests":
                    if self.final_container is None:
                        return Grade(
                            score=0.0,
                            success=False,
                            details={"error": "deep_swe_hidden_tests has no final container"},
                        )
                    grade_dir = Path(final.artifact_dir) / "grade" if final.artifact_dir else None
                    try:
                        reward = float(self.final_container.grade_deep_swe(patch, grade_dir))
                    except TypeError:
                        reward = float(self.final_container.grade_deep_swe(patch))
                    details = {"step_error": final.error} if final.error else {}
                    if grade_dir is not None:
                        details.update(
                            {
                                "public_test_log_ref": artifact_ref(grade_dir / "test_command.log"),
                                "hidden_grade_ref": artifact_ref(grade_dir / "reward_raw.txt"),
                            }
                        )
                    return Grade(score=reward, success=reward >= task.grader.success_threshold, details=details)
                return Grade(
                    score=0.0,
                    success=False,
                    details={"error": f"unsupported opencode_repo grader {task.grader.type!r}"},
                )
            from director.agentic.swebench_mini import grade_swesmith  # lazy: heavy agentic dep

            reward = float(grade_swesmith(instance, patch))
            details = {"step_error": final.error} if final.error else {}
            return Grade(score=reward, success=reward >= task.grader.success_threshold, details=details)
        except Exception as exc:  # noqa: BLE001 - live graders fail heterogeneously
            return Grade(score=0.0, success=False, details={"error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.close()

    def close(self) -> None:
        seen: set[int] = set()
        for container in self.owned:
            ident = id(container)
            if ident in seen:
                continue
            seen.add(ident)
            container.close()


@register_harness
class OpenCodeRepoAliasHarness(OpenCodeRepoHarness):
    """Backward-compatible task-harness name for older repo TaskSpecs."""

    name = "opencode_repo"


async def run_agentic_workflow(instance: dict, workflow, worker_slugs: list[str], key: str) -> dict:
    """Execute an ultra ``Workflow`` on a SWE-smith instance with container lineage.

    Returns ``{reward, final_diff, valid, steps:[{worker_id, slug, status, diff_len}]}``.
    """
    from director.agentic.swebench_mini import grade_swesmith  # lazy: heavy agentic dep

    image, iid = instance["image_name"], instance["instance_id"]
    problem = instance["problem_statement"].strip()

    containers: dict[int, OpenCodeContainer] = {}  # step index -> its workspace container
    diffs: dict[int, str] = {}
    owned: list[OpenCodeContainer] = []  # unique containers to clean up
    step_logs: list[dict] = []
    total_cost = 0.0

    try:
        for i, step in enumerate(workflow.steps):
            route = _opencode_route(worker_slugs[step.worker_id])
            access = list(step.access)
            if len(access) == 1:  # Rule B: continue in the predecessor's workspace
                container = containers[access[0]]
                prior_diffs, continuing = [], True
            else:  # Rule A (fresh) or Rule C (fresh + predecessor patches as artifacts)
                container = OpenCodeContainer(image, iid, oc_config=route["oc_config"])
                if not await asyncio.to_thread(container.start):
                    return {"reward": 0.0, "final_diff": "", "valid": False,
                            "steps": step_logs, "error": "container start/checkout failed"}
                owned.append(container)
                prior_diffs = [diffs[j] for j in access]
                continuing = bool(prior_diffs)

            slug = route["slug"]
            prompt = _step_prompt(problem, step.subtask, prior_diffs, continuing)
            run = await asyncio.to_thread(
                container.run_worker, slug, prompt,
                key_env=route["key_env"], oc_provider=route["oc_provider"], oc_config=route["oc_config"],
            )
            total_cost += float(run["cost"])
            diffs[i] = await asyncio.to_thread(container.diff)
            containers[i] = container
            step_logs.append(
                {
                    "worker_id": step.worker_id,
                    "slug": slug,
                    "status": run["status"],
                    "cost": float(run["cost"]),
                    "diff_len": len(diffs[i]),
                }
            )

        final_diff = diffs[len(workflow.steps) - 1]
        reward = 0.0
        if final_diff.strip():
            reward = float(
                await asyncio.to_thread(
                    grade_swesmith,
                    instance,
                    final_diff,
                    f"gate_oc_{uuid.uuid4().hex}",
                )
            )
        return {
            "reward": reward,
            "final_diff": final_diff,
            "valid": True,
            "steps": step_logs,
            "cost": total_cost,
        }
    finally:
        for container in owned:
            await asyncio.to_thread(container.close)
