"""Held-out SWE-bench Verified adapter smoke tests for ACRouter candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from .acrouter_candidates import _read_jsonl, _swebench_image_for_instance
from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    RepoRef,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)


SOURCE_NAME = "acrouter_swebench_verified"
SOURCE_VERSION = "ood176"


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _run_docker(args: list[str], *, timeout: int | None = None, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _image_exists(image: str) -> bool:
    proc = _run_docker(["image", "inspect", image], timeout=30)
    return proc.returncode == 0


def _safe_container_name(instance_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", instance_id)[:48].strip("-") or "swebench"
    return f"ultra-acr-{safe}-{int(time.time() * 1000) % 1000000}"


def _select_candidate(ready_jsonl: Path, instance_id: str | None = None) -> dict[str, Any]:
    rows = _read_jsonl(ready_jsonl)
    if not rows:
        raise ValueError(f"no candidates in {ready_jsonl}")
    if instance_id is None:
        return rows[0]
    for row in rows:
        if row.get("original_task_id") == instance_id or row.get("task_id") == instance_id:
            return row
    raise ValueError(f"candidate not found in {ready_jsonl}: {instance_id}")


def _load_swebench_verified_row(instance_id: str) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    for row in dataset:
        if row["instance_id"] == instance_id:
            return dict(row)
    raise ValueError(f"SWE-bench Verified instance not found: {instance_id}")


def _load_swebench_verified_rows(instance_ids: set[str]) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    out: dict[str, dict[str, Any]] = {}
    for row in dataset:
        if row["instance_id"] in instance_ids:
            out[row["instance_id"]] = dict(row)
        if len(out) == len(instance_ids):
            break
    missing = sorted(instance_ids - set(out))
    if missing:
        raise ValueError(f"SWE-bench Verified instances not found: {missing}")
    return out


def _redact_eval_log(text: str) -> str:
    """Remove answer/test-patch diff material from a SWE-bench eval log."""

    lines = text.splitlines()
    redacted: list[str] = []
    in_diff = False
    omitted = 0
    for line in lines:
        starts_diff = line.startswith("diff --git ") or line.startswith("--- a/") or line.startswith("+++ b/")
        if starts_diff:
            in_diff = True
            omitted += 1
            continue
        if in_diff:
            if (
                line.startswith("+ source")
                or line.startswith("+ python")
                or line.startswith("+ git checkout")
                or line.startswith("+ ./")
                or line.startswith(": ")
                or "Start Test Output" in line
            ):
                redacted.append(f"[redacted {omitted} diff/log lines]")
                in_diff = False
                omitted = 0
                redacted.append(line)
                continue
            if (
                line.startswith("+ ")
                or line.startswith("- ")
                or line.startswith("@@")
                or line.startswith("index ")
                or line.startswith("new file mode")
                or line.startswith("deleted file mode")
                or line.startswith("similarity index")
                or line.startswith("rename from")
                or line.startswith("rename to")
                or line.startswith("\\ No newline")
                or line.startswith(" ")
                or line.startswith("+")
                or line.startswith("-")
            ):
                omitted += 1
                continue
            redacted.append(f"[redacted {omitted} diff/log lines]")
            in_diff = False
            omitted = 0
        redacted.append(line)
    if in_diff:
        redacted.append(f"[redacted {omitted} diff/log lines]")
    return "\n".join(redacted) + ("\n" if text.endswith("\n") else "")


def _grade_patch_in_docker(
    instance: dict[str, Any],
    model_patch: str,
    *,
    image: str,
    log_dir: Path,
    eval_timeout: int,
    network: str,
) -> dict[str, Any]:
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance["instance_id"]
    result: dict[str, Any] = {
        "instance_id": instance_id,
        "image": image,
        "apply_ok": False,
        "resolved": None,
        "error": None,
        "elapsed_s": 0.0,
    }
    start_time = time.time()
    if not _image_exists(image):
        result["error"] = f"Docker image not found: {image}"
        return result

    test_spec = make_test_spec(instance)
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = log_dir / "test_output.raw.txt"
    redacted_log_path = log_dir / "test_output.redacted.txt"
    eval_error_path = log_dir / "eval_error.txt"
    container_name = _safe_container_name(instance_id)
    try:
        start = _run_docker(
            [
                "run",
                "-d",
                "--name",
                container_name,
                "--network",
                network,
                image,
                "tail",
                "-f",
                "/dev/null",
            ],
            timeout=300,
        )
        if start.returncode != 0:
            result["error"] = f"docker run failed: {(start.stderr or start.stdout).strip()[:500]}"
            return result

        if model_patch.strip():
            upload_patch = _run_docker(
                ["exec", "-i", container_name, "bash", "-lc", "cat > /tmp/model.patch"],
                input_text=model_patch,
                timeout=60,
            )
            if upload_patch.returncode != 0:
                result["error"] = f"patch upload failed: {(upload_patch.stderr or upload_patch.stdout).strip()[:500]}"
                return result
            apply = _run_docker(
                [
                    "exec",
                    container_name,
                    "bash",
                    "-lc",
                    "cd /testbed && (git apply --verbose /tmp/model.patch 2>&1 || "
                    "git apply -3 --verbose /tmp/model.patch 2>&1 || "
                    "patch --batch --fuzz=5 -p1 -i /tmp/model.patch 2>&1)",
                ],
                timeout=120,
            )
            result["apply_ok"] = apply.returncode == 0
            if not result["apply_ok"]:
                eval_error_path.write_text("patch apply failed\n" + apply.stdout + "\n" + apply.stderr)
                result["error"] = "patch apply failed"
                return result
        else:
            result["apply_ok"] = True

        upload = _run_docker(
            ["exec", "-i", container_name, "bash", "-lc", "cat > /tmp/eval.sh && chmod +x /tmp/eval.sh"],
            input_text=test_spec.eval_script,
            timeout=60,
        )
        if upload.returncode != 0:
            result["error"] = f"eval upload failed: {(upload.stderr or upload.stdout).strip()[:500]}"
            return result

        eval_proc = subprocess.run(
            ["docker", "exec", container_name, "bash", "/tmp/eval.sh"],
            capture_output=True,
            text=True,
            timeout=eval_timeout,
            check=False,
        )
        raw_log = eval_proc.stdout + "\n---STDERR---\n" + eval_proc.stderr
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(raw_log)
            tmp_path = Path(tmp.name)
        try:
            parsed = get_eval_report(
                test_spec=test_spec,
                prediction={
                    KEY_INSTANCE_ID: instance_id,
                    KEY_MODEL: "fugu-ultra-swebench-smoke",
                    KEY_PREDICTION: model_patch,
                },
                test_log_path=str(tmp_path),
                include_tests_status=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        redacted_log_path.write_text(_redact_eval_log(raw_log))
        result["raw_log_retained"] = False
        if raw_log_path.exists():
            raw_log_path.unlink()
        result["redacted_log_path"] = str(redacted_log_path)
        result["eval_returncode"] = eval_proc.returncode
        result["report"] = parsed.get(instance_id, parsed)
        result["resolved"] = bool(result["report"].get("resolved", False))
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout after {eval_timeout}s"
    except Exception as exc:  # noqa: BLE001 - smoke should report exact adapter issue
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _run_docker(["rm", "-f", container_name], timeout=60)
        result["elapsed_s"] = round(time.time() - start_time, 1)
    return result


def grade_swebench_verified_patch(
    *,
    instance_id: str,
    patch: str,
    image: str,
    log_dir: Path,
    eval_timeout: int = 1200,
    network: str = "none",
) -> dict[str, Any]:
    instance = _load_swebench_verified_row(instance_id)
    return _grade_patch_in_docker(
        instance,
        patch,
        image=image,
        log_dir=log_dir,
        eval_timeout=eval_timeout,
        network=network,
    )


def _task_id(instance_id: str) -> str:
    return f"{SOURCE_NAME}__{instance_id.replace('__', '__')}"


def _wall_time_for_difficulty(difficulty: str | None) -> int:
    normalized = (difficulty or "").strip().lower()
    if "1-4" in normalized or "1 - 4" in normalized:
        return 4 * 60 * 60
    if "<15" in normalized or "under 15" in normalized:
        return 30 * 60
    return 60 * 60


def materialize_swebench_ready_tasks(
    *,
    ready_jsonl: Path,
    out_jsonl: Path,
    report_out: Path | None = None,
    limit: int | None = None,
    swebench_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = _read_jsonl(ready_jsonl)
    if limit is not None:
        candidates = candidates[:limit]
    instance_ids = {str(row["original_task_id"]) for row in candidates}
    rows = swebench_rows or _load_swebench_verified_rows(instance_ids)
    specs: list[TaskSpec] = []
    for candidate in candidates:
        instance_id = str(candidate["original_task_id"])
        row = rows[instance_id]
        image = str((candidate.get("swebench") or {}).get("docker_image") or "")
        if not image:
            image = _swebench_image_for_instance(instance_id, "swebench/sweb.eval.x86_64")
        swebench_meta = candidate.get("swebench") or {}
        difficulty = str(swebench_meta.get("difficulty") or "")
        wall_time_seconds = _wall_time_for_difficulty(difficulty)
        problem = str(row.get("problem_statement") or candidate.get("prompt") or "")
        repo = str(row.get("repo") or "")
        base_commit = str(row.get("base_commit") or "")
        opencode_instance = {
            "image_name": image,
            "instance_id": "",
            "swebench_instance_id": instance_id,
            "problem_statement": problem,
            "testbed": "/testbed",
            "grader": "swebench_verified",
            "test_command": "official_swebench_eval_script",
            "task_id": instance_id,
        }
        specs.append(
            TaskSpec(
                task_id=_task_id(instance_id),
                capability="agentic_coding",
                source=SourceRef(
                    name=SOURCE_NAME,
                    version=SOURCE_VERSION,
                    policy="pool_only",
                    url_or_ref=str(ready_jsonl),
                    license="see-source",
                ),
                input=TaskInput(
                    messages=[{"role": "user", "content": problem}],
                    assets=[
                        {"opencode_instance": opencode_instance},
                        {
                            "acrouter_disagreement": {
                                "candidate_id": candidate.get("candidate_id"),
                                "task_id": candidate.get("task_id"),
                                "success_count": candidate.get("success_count"),
                                "model_count": candidate.get("model_count"),
                                "disagreement_balance": candidate.get("disagreement_balance"),
                                "difficulty": difficulty,
                            }
                        },
                    ],
                    repo=RepoRef(url=f"https://github.com/{repo}", base_commit=base_commit)
                    if repo and base_commit
                    else None,
                ),
                environment=EnvironmentSpec(
                    harness="opencode",
                    image=image,
                    cpu_limit=2,
                    memory_mb=8192,
                    disk_mb=20480,
                    wall_time_seconds=wall_time_seconds,
                ),
                grader=GraderSpec(
                    type="swebench_verified_hidden_tests",
                    command=["official_swebench_eval_script"],
                    success_threshold=1.0,
                ),
                splitting=SplittingSpec(
                    group_id=repo or instance_id,
                    split="pool_validation",
                    contamination_group=f"swebench_verified/{repo or instance_id}",
                ),
                metadata=TaskMetadata(
                    domain="software_engineering",
                    subdomain="bug_fixing",
                    tags=[
                        "acrouter",
                        "ood176",
                        "swebench_verified",
                        "held_out",
                        "pool_validation",
                        "disagreement_mined",
                    ],
                    requires_tools=True,
                    estimated_worker_calls=3,
                ),
            )
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "version": "acrouter_swebench_ready_tasks_v1",
        "ready_jsonl": str(ready_jsonl),
        "out_jsonl": str(out_jsonl),
        "task_count": len(specs),
        "source": SOURCE_NAME,
        "policy": "pool_only",
        "split": "pool_validation",
        "grpo_ready": 0,
        "task_ids": [spec.task_id for spec in specs],
        "instance_ids": [str(row["original_task_id"]) for row in candidates],
        "grader_type": "swebench_verified_hidden_tests",
        "wall_time_by_task": {
            spec.task_id: spec.environment.wall_time_seconds
            for spec in specs
        },
    }
    if report_out is not None:
        _write_json(report_out, report)
    return report


def run_swebench_ready_smoke(
    *,
    ready_jsonl: Path,
    out: Path,
    log_dir: Path,
    instance_id: str | None = None,
    patch_source: str = "gold",
    image_prefix: str = "swebench/sweb.eval.x86_64",
    eval_timeout: int = 1200,
    network: str = "none",
) -> dict[str, Any]:
    candidate = _select_candidate(ready_jsonl, instance_id)
    instance_id = str(candidate["original_task_id"])
    instance = _load_swebench_verified_row(instance_id)
    if patch_source == "gold":
        patch = str(instance.get("patch") or "")
    elif patch_source == "empty":
        patch = ""
    else:
        raise ValueError(f"unsupported patch_source: {patch_source}")
    image = str((candidate.get("swebench") or {}).get("docker_image") or "")
    if not image:
        image = _swebench_image_for_instance(instance_id, image_prefix)

    grade = _grade_patch_in_docker(
        instance,
        patch,
        image=image,
        log_dir=log_dir / instance_id,
        eval_timeout=eval_timeout,
        network=network,
    )
    report = {
        "version": "acrouter_swebench_verified_smoke_v1",
        "candidate_id": candidate.get("candidate_id"),
        "task_id": candidate.get("task_id"),
        "instance_id": instance_id,
        "source_dataset": "princeton-nlp/SWE-bench_Verified",
        "policy": candidate.get("permitted_use"),
        "patch_source": patch_source,
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else None,
        "patch_chars": len(patch),
        "gold_patch_retained": False,
        "image": image,
        "eval_timeout": eval_timeout,
        "network": network,
        "grade": grade,
        "success": bool(grade.get("resolved")),
        "grpo_ready": False,
        "notes": "Held-out adapter smoke only; do not use gold patch or SWE-bench Verified OOD row for GRPO training.",
    }
    _write_json(out, report)
    return report
