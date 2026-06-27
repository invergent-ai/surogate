"""SWE-Bench environment: an AgentEnv backed by a SWE-Bench instance Docker container.

Design (mirrors mini-swe-agent + the official grader):
  - reset(): start a container from the instance's SWE-Bench image (repo checked out at
    base_commit under /testbed) and return the problem statement.
  - step(cmd): ``docker exec`` the bash command in /testbed; return truncated output.
  - evaluate(): take the agent's ``git diff`` as the model patch and score it with the
    OFFICIAL swebench harness (apply patch + test patch, run FAIL_TO_PASS/PASS_TO_PASS).

Requires ``pip install swebench`` and Docker, plus the instance images (built once via
swebench). Heavy: images are large and grading spins its own container. The routing /
rollout / fitness logic is exercised offline via ScriptedEnv; this is the real harness.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .env import StepResult

TESTBED = "/testbed"


def load_swebench(dataset: str = "princeton-nlp/SWE-bench_Verified", split: str = "test", limit=None, shuffle=False, seed=0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    rows = list(ds)
    if shuffle:
        import random

        random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def load_swebench_pro(dataset: str = "ScaleAI/SWE-bench_Pro", split: str = "test", limit=None, shuffle=False, seed=0) -> list[dict]:
    """Load SWE-Bench Pro instances (multi-language: go/python/js/ts).

    Pro has its own schema (``dockerhub_tag``, lowercase ``fail_to_pass``/``pass_to_pass``,
    ``before_repo_set_cmd``, ``selected_test_files_to_run``). Faithful grading requires
    ScaleAI's official Pro harness (per-language test runners + log parsing); this loader
    only normalizes the rows so they can be driven by that harness via an AgentEnv.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    rows = list(ds)
    if shuffle:
        import random

        random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def _instance_image(instance: dict, namespace: str | None = "swebench") -> str:
    """Resolve the SWE-Bench instance image tag via swebench's own spec helper.

    With ``namespace`` set (default "swebench"), this returns the dockerhub-pullable
    name (e.g. ``swebench/sweb.eval.x86_64.<sanitized>:latest``), so ``docker run``
    auto-pulls the prebuilt image. Pass ``namespace=None`` to use a locally-built image.
    """
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(instance, namespace=namespace).instance_image_key


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@dataclass
class SWEBenchEnv:
    instance: dict
    dataset: str = "princeton-nlp/SWE-bench_Verified"
    run_id: str = field(default_factory=lambda: f"director_{uuid.uuid4().hex[:8]}")
    step_timeout: float = 120.0
    image: str | None = None
    namespace: str | None = "swebench"
    _cid: str | None = None

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> str:
        image = self.image or _instance_image(self.instance, namespace=self.namespace)
        name = f"director-{self.instance['instance_id'].replace('/', '_')}-{uuid.uuid4().hex[:6]}"
        proc = _run(["docker", "run", "-d", "--name", name, image, "sleep", "infinity"])
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed for {image}: {proc.stderr.strip()}")
        self._cid = name
        # ensure a clean tree at base_commit so the agent's diff is its own work
        self._exec("git -C %s reset --hard && git -C %s clean -fd" % (TESTBED, TESTBED))
        return self.instance["problem_statement"]

    def _exec(self, command: str) -> tuple[int, str]:
        assert self._cid is not None, "call reset() first"
        proc = _run(
            ["docker", "exec", self._cid, "bash", "-lc", f"cd {TESTBED} && {command}"],
            timeout=self.step_timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr)

    def step(self, command: str) -> StepResult:
        try:
            _rc, out = self._exec(command)
        except subprocess.TimeoutExpired:
            return StepResult(observation=f"[timed out after {self.step_timeout}s]")
        return StepResult(observation=out)

    def model_patch(self) -> str:
        _rc, out = self._exec("git add -A >/dev/null 2>&1; git diff --cached")
        return out

    # -- grading via the official harness -----------------------------------
    def evaluate(self) -> float:
        patch = self.model_patch()
        if not patch.strip():
            return 0.0
        iid = self.instance["instance_id"]
        with tempfile.TemporaryDirectory() as td:
            pred_path = os.path.join(td, "preds.jsonl")
            with open(pred_path, "w") as f:
                f.write(json.dumps({
                    "instance_id": iid,
                    "model_name_or_path": "director",
                    "model_patch": patch,
                }) + "\n")
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", self.dataset,
                "--predictions_path", pred_path,
                "--max_workers", "1",
                "--run_id", self.run_id,
                "--instance_ids", iid,
                "--cache_level", "env",
            ]
            _run(cmd, timeout=self.step_timeout * 20)
            report = f"director.{self.run_id}.json"
            if not os.path.exists(report):
                return 0.0
            with open(report) as f:
                data = json.load(f)
            try:
                os.remove(report)
            except OSError:
                pass
            return 1.0 if iid in set(data.get("resolved_ids", [])) else 0.0

    def close(self) -> None:
        if self._cid:
            _run(["docker", "rm", "-f", self._cid])
            self._cid = None


def build_swebench_factories(
    instances: list[dict], dataset: str = "princeton-nlp/SWE-bench_Verified", step_timeout: float = 120.0
) -> list[Callable[[], SWEBenchEnv]]:
    """One fresh-env factory per instance (each rollout gets its own container)."""
    return [
        (lambda inst=inst: SWEBenchEnv(instance=inst, dataset=dataset, step_timeout=step_timeout))
        for inst in instances
    ]
