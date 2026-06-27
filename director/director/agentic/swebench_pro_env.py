"""SWE-Bench Pro environment (ScaleAI).

Pro differs from Verified: multi-language (go/py/js/ts), its own images
(``jefzda/sweap-images:{tag}``), ENTRYPOINT ``/bin/bash``, repo at ``/app``, and grading
via ScaleAI's clone-and-run harness (github.com/scaleapi/SWE-bench_Pro-os), which uses
per-instance ``run_scripts/{iid}/`` + ``parser.py`` for per-language test parsing.

The agent loop (reset/step/model_patch) is validated here directly; faithful grading is
delegated to the official harness (``evaluate`` shells out to ``swe_bench_pro_eval.py
--use_local_docker``). Set ``harness_dir`` to a clone of that repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .env import StepResult

TESTBED = "/app"
DOCKERHUB_USER = "jefzda"


def instance_image(instance: dict, dockerhub_username: str = DOCKERHUB_USER) -> str:
    """Pullable image name. Mirrors helper_code/image_uri.get_dockerhub_image_uri."""
    uid = instance["instance_id"]
    uid = uid if uid.startswith("instance_") else f"instance_{uid}"
    repo_base, repo_name_only = instance["repo"].lower().split("/")
    hsh = uid.replace("instance_", "")
    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in instance["repo"].lower() and "element-web" in instance["repo"].lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]
    tag = f"{repo_base}.{repo_name_only}-{hsh}"[:128]
    return f"{dockerhub_username}/sweap-images:{tag}"


def _run(cmd: list[str], timeout: float | None = None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def _as_list(v):
    """fail_to_pass/pass_to_pass arrive as a JSON/py-list string or a real list."""
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        try:
            import ast

            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []


@dataclass
class SWEBenchProEnv:
    instance: dict
    dockerhub_username: str = DOCKERHUB_USER
    harness_dir: str | None = None  # clone of scaleapi/SWE-bench_Pro-os (for grading)
    step_timeout: float = 120.0
    image: str | None = None
    _cid: str | None = None

    def reset(self) -> str:
        image = self.image or instance_image(self.instance, self.dockerhub_username)
        name = f"director-pro-{uuid.uuid4().hex[:8]}"
        # ENTRYPOINT is /bin/bash, so override it to keep the container alive.
        proc = _run(["docker", "run", "-d", "--entrypoint", "sleep", "--name", name, image, "infinity"])
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed for {image}: {proc.stderr.strip()}")
        self._cid = name
        if self.instance.get("before_repo_set_cmd"):
            self._exec(self.instance["before_repo_set_cmd"])
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

    def evaluate(self) -> float:
        """Grade this env's own container patch (git diff) with ScaleAI's harness."""
        return self.grade_patch(self.model_patch())

    def grade_patch(self, patch: str) -> float:
        """Grade a GIVEN patch with ScaleAI's official harness (faithful, multi-lang). Lets a
        patch produced by another agent (e.g. mini-swe-agent) be graded without this env's
        container. Requires ``harness_dir`` (clone of scaleapi/SWE-bench_Pro-os) + Docker.
        """
        if not self.harness_dir:
            raise RuntimeError(
                "SWE-Bench Pro grading needs harness_dir = clone of scaleapi/SWE-bench_Pro-os"
            )
        if not patch or not patch.strip():
            return 0.0
        iid = self.instance["instance_id"]
        out_dir = os.path.join(self.harness_dir, f"director_out_{uuid.uuid4().hex[:6]}")
        os.makedirs(out_dir, exist_ok=True)
        patch_path = os.path.join(out_dir, "patches.json")
        with open(patch_path, "w") as f:
            json.dump([{"instance_id": iid, "patch": patch, "prefix": "pred"}], f)
        csv_path = self._write_sample_csv(out_dir)
        _run(
            [
                "python", "swe_bench_pro_eval.py",
                f"--raw_sample_path={csv_path}",
                f"--patch_path={patch_path}",
                f"--output_dir={out_dir}",
                "--scripts_dir=run_scripts",
                "--num_workers=1",
                f"--dockerhub_username={self.dockerhub_username}",
                "--use_local_docker",
            ],
            timeout=self.step_timeout * 30,
            cwd=self.harness_dir,
        )
        return self._parse_resolved(out_dir, iid)

    def _write_sample_csv(self, out_dir: str) -> str:
        import csv

        i = self.instance
        path = os.path.join(out_dir, "sample.csv")
        cols = {
            "instance_id": i["instance_id"],
            "before_repo_set_cmd": i.get("before_repo_set_cmd", ""),
            "selected_test_files_to_run": i.get("selected_test_files_to_run", ""),
            "base_commit": i.get("base_commit", ""),
            "FAIL_TO_PASS": i.get("fail_to_pass", ""),
            "PASS_TO_PASS": i.get("pass_to_pass", ""),
        }
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cols))
            w.writeheader()
            w.writerow(cols)
        return path

    def _parse_resolved(self, out_dir: str, iid: str) -> float:
        # Harness writes per-instance {prefix}_output.json = {"tests": [{name, status}]}.
        # Resolved iff (fail_to_pass | pass_to_pass) are all PASSED (official logic).
        out_json = os.path.join(out_dir, iid, "pred_output.json")
        if not os.path.exists(out_json):
            return 0.0
        try:
            with open(out_json) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return 0.0
        passed = {t["name"] for t in data.get("tests", []) if t.get("status") == "PASSED"}
        f2p = set(_as_list(self.instance.get("fail_to_pass", "[]")))
        p2p = set(_as_list(self.instance.get("pass_to_pass", "[]")))
        return 1.0 if (f2p | p2p) <= passed else 0.0

    def close(self) -> None:
        if self._cid:
            _run(["docker", "rm", "-f", self._cid])
            self._cid = None


def build_swebench_pro_factories(
    instances: list[dict], harness_dir: str | None = None, dockerhub_username: str = DOCKERHUB_USER,
    step_timeout: float = 120.0,
) -> list[Callable[[], SWEBenchProEnv]]:
    return [
        (lambda inst=inst: SWEBenchProEnv(
            instance=inst, harness_dir=harness_dir,
            dockerhub_username=dockerhub_username, step_timeout=step_timeout,
        ))
        for inst in instances
    ]
