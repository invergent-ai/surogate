"""Task-level build-and-debug orchestration over a pier-style harness.

The Fugu report's validated CODING topology is not a per-turn tree — it is
sequential agent RUNS over the same task (§4.4): a builder produces a patch,
an independent verifier reviews it against the task's checks, and the builder
applies the fixes. For container harnesses (pier/Harbor), each run starts from
the task's base commit, so state is carried BETWEEN runs by augmenting the
next run's instruction with the prior patch and review — the run applies the
inherited patch first, then does its own work. The final run's committed state
is therefore base + patch + fixes, which is exactly what `pre_artifacts.sh`
captures for the verifier container.

The pier invocation is injected (`runner`), so the orchestration logic is
offline-testable and harness-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# One runner call = one full agent run of one task; returns the run's reward
# record: {"reward": float|None, "patch": str, "notes": str}
class TaskRunner(Protocol):
    def __call__(self, *, task_name: str, instruction: str, run_label: str) -> dict: ...


BUILDER_PREFIX = (
    "You are the BUILDER. Complete the task below. Work on a new branch and "
    "commit everything, as the task requires.\n\n"
)

INHERIT_TEMPLATE = (
    "A previous attempt produced the patch below. FIRST apply it exactly "
    "(`git apply` the diff, resolve trivially if needed, commit), THEN do "
    "your part.\n\n--- PRIOR PATCH ---\n{patch}\n--- END PRIOR PATCH ---\n\n"
)

VERIFIER_TEMPLATE = (
    "You are the VERIFIER for the task below. A builder produced the patch "
    "included above, which you have applied. Do NOT rewrite the solution. "
    "Run the repository's relevant tests/checks, inspect the diff against the "
    "task requirements, and WRITE your findings to a file named "
    "REVIEW.md at the repository root (list each concrete defect, or state "
    "'NO DEFECTS FOUND'), then commit REVIEW.md.\n\n"
)

FIXER_TEMPLATE = (
    "You are the BUILDER again. Your earlier patch (applied above) was "
    "reviewed; the review follows. Fix every defect it lists, re-run the "
    "relevant checks, and commit. If the review found no defects, verify the "
    "checks pass and commit any final cleanup.\n\n--- REVIEW ---\n{review}\n"
    "--- END REVIEW ---\n\n"
)


@dataclass
class BuildDebugDriver:
    """Runs builder -> verifier -> fixer over one harness task."""

    runner: TaskRunner
    log: list[dict] = field(default_factory=list)

    def run_task(self, task_name: str, instruction: str) -> dict[str, Any]:
        # 1) builder attempt
        build = self.runner(
            task_name=task_name,
            instruction=BUILDER_PREFIX + instruction,
            run_label="build",
        )
        self.log.append({"task": task_name, "run": "build", **_meta(build)})
        patch = (build.get("patch") or "").strip()
        if not patch:
            # nothing to review — the builder's outcome is the task's outcome
            return {"reward": build.get("reward"), "runs": 1, "path": "build-only"}

        # 2) independent verifier over the applied patch
        verify_instruction = (
            INHERIT_TEMPLATE.format(patch=patch) + VERIFIER_TEMPLATE + instruction
        )
        verify = self.runner(
            task_name=task_name, instruction=verify_instruction, run_label="verify"
        )
        self.log.append({"task": task_name, "run": "verify", **_meta(verify)})
        review = (verify.get("notes") or "").strip()

        # 3) builder applies fixes with access to both
        fix_instruction = (
            INHERIT_TEMPLATE.format(patch=patch)
            + FIXER_TEMPLATE.format(review=review or "NO REVIEW PRODUCED")
            + instruction
        )
        fix = self.runner(
            task_name=task_name, instruction=fix_instruction, run_label="fix"
        )
        self.log.append({"task": task_name, "run": "fix", **_meta(fix)})

        # The FIX run's verifier verdict is the task outcome; if the fix run
        # failed to produce a scoreable result, fall back to the build's.
        reward = fix.get("reward")
        if reward is None:
            reward = build.get("reward")
            path = "fix-unscored:build-fallback"
        else:
            path = "build-verify-fix"
        return {"reward": reward, "runs": 3, "path": path}


def _meta(run: dict) -> dict:
    return {
        "reward": run.get("reward"),
        "patch_chars": len(run.get("patch") or ""),
        "notes_chars": len(run.get("notes") or ""),
    }


def load_task_instruction(tasks_root: Path, task_name: str) -> str:
    return (tasks_root / task_name / "instruction.md").read_text()


def pier_runner(
    *,
    pier_bin: str,
    tasks_root: Path,
    jobs_root: Path,
    agent: str = "mini-swe-agent",
    model: str | dict[str, str],
    env: dict[str, str] | None = None,
    run_cmd: Callable[..., Any] | None = None,
) -> TaskRunner:
    """Real runner: one pier invocation per call, on a per-run task copy.

    `model` may be a single id, or a mapping run_label -> model id so each
    workflow position is owned by the conductor-assigned worker (Ultra
    semantics: one worker owns each step's entire agent loop). The task copy
    gets the augmented instruction written to instruction.md; everything else
    (image, tests, pre_artifacts) is untouched, so rewards remain the task's
    own held-out verification.
    """
    import shutil
    import subprocess

    def call(*, task_name: str, instruction: str, run_label: str) -> dict:
        run_model = model if isinstance(model, str) else model[run_label]
        job = f"{task_name[:40]}-{run_label}"
        staged = jobs_root / "staged" / job / task_name
        if staged.parent.exists():
            shutil.rmtree(staged.parent)
        shutil.copytree(tasks_root / task_name, staged)
        (staged / "instruction.md").write_text(instruction)
        # A stale job dir (e.g. from an interrupted earlier run) makes pier
        # treat the job as already attempted and return its old result
        # instantly — wipe it so every invocation actually runs.
        if (jobs_root / job).exists():
            shutil.rmtree(jobs_root / job)

        cmd = [
            pier_bin, "run", "-p", str(staged.parent), "--agent", agent,
            "--model", run_model, "--job-name", job, "--jobs-dir", str(jobs_root),
        ]
        runner = run_cmd or subprocess.run
        runner(cmd, env=env, capture_output=True, text=True, timeout=7200)

        # Reclaim this run's docker network. Pier leaves one bridge network per
        # trial behind; Docker's default address pool holds only ~31, so a few
        # interrupted runs exhaust it and every later `compose up` fails with
        # "all predefined address pools have been fully subnetted". Scoped to
        # THIS job's project name — never a blanket prune (other users' Docker
        # resources must not be touched).
        try:
            subprocess.run(
                ["docker", "network", "prune", "-f", "--filter",
                 f"name={job.lower()}"],
                capture_output=True, timeout=60,
            )
        except Exception:  # noqa: BLE001 — cleanup must never fail the run
            pass

        result_file = jobs_root / job / "result.json"
        reward = None
        patch = ""
        if result_file.exists():
            payload = json.loads(result_file.read_text())
            reward = _extract_reward(payload)
        # the captured artifact patch, if the harness stored it
        for candidate in (jobs_root / job).rglob("model.patch"):
            patch = candidate.read_text()
            break
        notes = ""
        for candidate in (jobs_root / job).rglob("REVIEW.md"):
            notes = candidate.read_text()
            break
        return {"reward": reward, "patch": patch, "notes": notes}

    return call


def _extract_reward(payload: Any) -> float | None:
    """Find the verifier reward in a pier/harbor result payload."""
    if isinstance(payload, dict):
        if "reward" in payload and isinstance(payload["reward"], (int, float)):
            return float(payload["reward"])
        for value in payload.values():
            found = _extract_reward(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _extract_reward(value)
            if found is not None:
                return found
    return None
