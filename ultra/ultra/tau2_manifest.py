"""tau2-telecom training-manifest exporter (office-task mixture).

Emits TaskSpec rows for the tau2 telecom SOLO lane from ``telecom_full``
(2285 generated tickets) while sealing the published 114-task ``telecom``
set for evaluation — those ids must never enter a training manifest, and
``export`` enforces that as a hard invariant rather than a convention.

Task ids are ``tau2__telecom_full__<sha1[:12]>`` of the tau2 task id: the
upstream ids embed scenario text (brackets, pipes) and are unwieldy as keys,
and positional indices are unstable across dataset versions. The original id
travels in the grader payload, which is what ``Tau2SoloHarness`` resolves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DOMAIN = "telecom"
TRAIN_TASK_SET = "telecom_full"
SEALED_TASK_SET = "telecom"  # published tau2-telecom benchmark: eval only
CAPABILITY = "office_telecom"
# Gold action counts run 1-11 (median 6, p99 10); 20 leaves room for
# diagnostic reads without paying for 30 calls on every rollout.
DEFAULT_MAX_TURNS = 20

_REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_TAU2 = _REPO_ROOT / "director" / "vendor" / "tau2_bench"


def _ensure_tau2() -> None:
    if "tau2" not in sys.modules:
        os.environ.setdefault("TAU2_DATA_DIR", str(VENDORED_TAU2 / "data"))
    import tau2  # noqa: F401


def _source_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(VENDORED_TAU2), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def spec_task_id(tau2_task_id: str) -> str:
    digest = hashlib.sha1(tau2_task_id.encode()).hexdigest()[:12]
    return f"tau2__{TRAIN_TASK_SET}__{digest}"


def sealed_task_ids() -> set[str]:
    _ensure_tau2()
    from tau2.registry import registry

    return {t.id for t in registry.get_tasks_loader(SEALED_TASK_SET)()}


def train_tasks() -> list[Any]:
    _ensure_tau2()
    from tau2.registry import registry

    sealed = sealed_task_ids()
    return [t for t in registry.get_tasks_loader(TRAIN_TASK_SET)() if t.id not in sealed]


def build_taskspec(task: Any, *, source_commit: str | None, max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, Any]:
    task_id = spec_task_id(task.id)
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": CAPABILITY,
        "source": {
            "name": "tau2_telecom_full",
            "url_or_ref": "https://github.com/sierra-research/tau2-bench"
                          + (f"@{source_commit}" if source_commit else ""),
            "version": f"sierra-{source_commit}" if source_commit else "sierra-unpinned",
            "license": "MIT",
            "policy": "train_allowed",
            "source_commit": source_commit,
        },
        "input": {
            # The ticket travels in the prompt so the conductor can route on
            # what the task actually is; the solo worker receives the same
            # ticket in its tau2 system prompt, so this leaks nothing.
            "messages": [
                {
                    "role": "user",
                    "content": "Customer support ticket (telecom). Resolve it by "
                               "operating the support tools and the customer's "
                               f"device tools.\n\n{(task.ticket or '').strip()}",
                }
            ],
            "context_documents": [],
            "assets": [],
            "repo": None,
            "tools": [],
        },
        "environment": {
            "harness": "tau2_solo",
            "image": None,
            "cpu_limit": None,
            "memory_mb": None,
            "disk_mb": None,
            "network_policy": "model-relay-only",
            "wall_time_seconds": 1200,
        },
        "grader": {
            "type": "tau2_programmatic",
            "command": None,
            "expected_answer": {
                "domain": DOMAIN,
                "task_set": TRAIN_TASK_SET,
                "tau2_task_id": task.id,
                "max_turns": max_turns,
                "evaluation_type": "ALL",
            },
            "score_range": [0.0, 1.0],
            "success_threshold": 1.0,
            "deterministic": True,
        },
        "splitting": {
            "split": "grpo_train",
            "group_id": "tau2_telecom_full_train",
            "contamination_group": f"tau2_telecom/{task.id}",
        },
        "metadata": {
            "domain": CAPABILITY,
            "subdomain": "tau2_telecom_solo",
            "difficulty_estimate": None,
            "estimated_worker_calls": max_turns,
            "requires_tools": True,
            "requires_long_context": False,
            "tags": ["tau2", "telecom", "solo", "train", "real_tools", "programmatic_reward"],
        },
    }


def export(out_dir: Path, *, max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, Any]:
    """Write tau2_telecom_taskspecs.jsonl + the sealed-eval id list."""
    sealed = sealed_task_ids()
    tasks = train_tasks()
    commit = _source_commit()

    rows = [build_taskspec(t, source_commit=commit, max_turns=max_turns) for t in tasks]
    leaked = [r for r in rows if r["grader"]["expected_answer"]["tau2_task_id"] in sealed]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} sealed tau2-telecom eval tasks reached the training "
            "manifest — refusing to write it"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "tau2_telecom_taskspecs.jsonl"
    manifest_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    sealed_path = out_dir / "tau2_telecom_sealed_eval_ids.json"
    sealed_path.write_text(json.dumps(
        {
            "task_set": SEALED_TASK_SET,
            "policy": "eval_only_never_train",
            "task_ids": sorted(sealed),
        },
        indent=1,
    ))
    return {
        "train_tasks": len(rows),
        "sealed_eval_tasks": len(sealed),
        "manifest": str(manifest_path),
        "sealed_ids": str(sealed_path),
        "source_commit": commit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    args = parser.parse_args(argv)
    summary = export(args.out_dir, max_turns=args.max_turns)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
