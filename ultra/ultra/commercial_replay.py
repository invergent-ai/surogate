"""Build offline replay/SFT artifacts from completed commercial rollouts.

The output is intentionally conservative:

* replay JSONL keeps complete commercial-inclusive rollout evidence, including
  failures, with original workflow/execution payloads preserved.
* SFT JSONL is emitted only for successful commercial-inclusive workflows and
  remaps worker IDs to the local worker table shown in the prompt.

No provider calls are made here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .scaffold_tournament import canonical_workers

REPO_ROOT = Path(__file__).resolve().parents[2]

COMMERCIAL_NAME_MARKERS = ("gpt", "opus", "gemini", "claude", "codex")
TRAINABLE_OUTCOMES = {
    "valid_correct_trainable",
    "valid_incorrect_trainable",
    "budget_exhausted_trainable",
}
SUCCESS_OUTCOME = "valid_correct_trainable"

SYSTEM_PROMPT = """You are the Fugu-Ultra Conductor.

Choose a short workflow over the allowed workers for this task. Return only JSON:

{"steps":[{"worker_id":0,"subtask":"...","access":[],"budget":"medium"}]}

Rules:
- worker_id is an integer from the allowed worker table in this prompt.
- Use 1 to 3 steps.
- access may only list earlier step indexes.
- budget must be one of "short", "medium", "long", or "max".
- Do not answer the user task yourself. Route work to workers through the workflow.
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _resolve_path(raw: str | None, *, report_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [
        REPO_ROOT / path,
        report_dir / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _worker_catalog_by_id() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for worker in canonical_workers():
        dumped = worker.model_dump()
        out[int(worker.worker_id)] = dumped
    return out


def _worker_catalog_by_name() -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in _worker_catalog_by_id().values()}


def is_commercial_worker(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in COMMERCIAL_NAME_MARKERS)


def _workflow_worker_names(rollout: dict[str, Any], by_id: dict[int, dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for step in (rollout.get("workflow") or {}).get("steps") or []:
        try:
            worker_id = int(step["worker_id"])
        except (KeyError, TypeError, ValueError):
            continue
        worker = by_id.get(worker_id)
        names.append(str(worker.get("name") if worker else f"worker_{worker_id}"))
    return names


def _row_worker_names(row: dict[str, Any], job: dict[str, Any] | None, rollout: dict[str, Any]) -> list[str]:
    names = row.get("worker_names") or (job or {}).get("worker_names") or _workflow_worker_names(
        rollout, _worker_catalog_by_id()
    )
    return [str(name) for name in names]


def _is_complete_replay_record(row: dict[str, Any], rollout: dict[str, Any]) -> bool:
    outcome = row.get("outcome_class") or rollout.get("outcome_class")
    return (
        bool(rollout.get("workflow"))
        and bool(rollout.get("execution"))
        and rollout.get("reward") is not None
        and outcome in TRAINABLE_OUTCOMES
        and (row.get("valid_for_training", True) is not False)
        and (rollout.get("valid_for_training", True) is not False)
    )


def _load_job_map(report: dict[str, Any], *, report_dir: Path) -> dict[str, dict[str, Any]]:
    jobs_path = _resolve_path(report.get("jobs_jsonl"), report_dir=report_dir)
    if not jobs_path or not jobs_path.exists():
        return {}
    return {str(row.get("job_id")): row for row in _read_jsonl(jobs_path) if row.get("job_id")}


def _load_task_index(path: Path, cache: dict[Path, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    if path not in cache:
        cache[path] = {str(row.get("task_id")): row for row in _read_jsonl(path) if row.get("task_id")}
    return cache[path]


def _task_for_record(
    row: dict[str, Any],
    job: dict[str, Any] | None,
    *,
    report_dir: Path,
    task_cache: dict[Path, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    task_jsonl = (job or {}).get("task_jsonl")
    task_path = _resolve_path(task_jsonl, report_dir=report_dir)
    if not task_path or not task_path.exists():
        return None
    index = _load_task_index(task_path, task_cache)
    for key in (row.get("task_id"), row.get("source_task_id"), (job or {}).get("source_task_id")):
        if key and str(key) in index:
            return index[str(key)]
    return None


def _messages_text(task: dict[str, Any], *, max_chars: int = 14000) -> str:
    parts: list[str] = []
    for msg in (task.get("input") or {}).get("messages") or []:
        role = msg.get("role", "message")
        content = str(msg.get("content") or "")
        parts.append(f"[{role}]\n{content}")
    repo = (task.get("input") or {}).get("repo")
    if repo:
        parts.append(
            "[repo]\n"
            f"url={repo.get('url')}\n"
            f"base_commit={repo.get('base_commit')}\n"
            f"subdirectory={repo.get('subdirectory')}"
        )
    docs = (task.get("input") or {}).get("context_documents") or []
    if docs:
        doc_lines = []
        for i, doc in enumerate(docs, start=1):
            if isinstance(doc, dict):
                title = doc.get("title") or doc.get("id") or f"document-{i}"
                content = doc.get("text") or doc.get("content") or json.dumps(doc, sort_keys=True)
            else:
                title = f"document-{i}"
                content = str(doc)
            doc_lines.append(f"- {title}: {str(content)[:600]}")
        parts.append("[context documents]\n" + "\n".join(doc_lines))
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated for conductor prompt]"
    return text


def _compact_workflow(
    rollout: dict[str, Any],
    allowed_worker_names: list[str],
    *,
    by_id: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name_to_local = {name: i for i, name in enumerate(allowed_worker_names)}
    allowed_workers = []
    by_name = _worker_catalog_by_name()
    for local_id, name in enumerate(allowed_worker_names):
        catalog_row = by_name.get(name, {})
        allowed_workers.append(
            {
                "worker_id": local_id,
                "name": name,
                "backend": catalog_row.get("backend"),
                "model": catalog_row.get("model"),
                "role_prior": catalog_row.get("role_prior") or [],
            }
        )

    steps = []
    for step in (rollout.get("workflow") or {}).get("steps") or []:
        original_id = int(step["worker_id"])
        worker = by_id.get(original_id)
        name = str(worker.get("name") if worker else f"worker_{original_id}")
        if name not in name_to_local:
            name_to_local[name] = len(allowed_workers)
            allowed_workers.append({"worker_id": name_to_local[name], "name": name})
        compact = {
            "worker_id": name_to_local[name],
            "subtask": step.get("subtask") or "",
            "access": list(step.get("access") or []),
            "budget": step.get("budget") or "medium",
        }
        steps.append(compact)
    return {"steps": steps}, allowed_workers


def _sft_messages(task: dict[str, Any], allowed_workers: list[dict[str, Any]]) -> list[dict[str, str]]:
    worker_lines = []
    for row in allowed_workers:
        roles = ", ".join(str(role) for role in row.get("role_prior") or [])
        worker_lines.append(
            f"{row['worker_id']}: {row['name']} | backend={row.get('backend')} | "
            f"model={row.get('model')} | roles={roles}"
        )

    user = "\n\n".join(
        [
            f"Task ID: {task.get('task_id')}",
            f"Capability: {task.get('capability')}",
            f"Task harness: {(task.get('environment') or {}).get('harness')}",
            "Allowed workers:\n" + "\n".join(worker_lines),
            "Task prompt:\n" + _messages_text(task),
        ]
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_commercial_replay(manifest_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = _worker_catalog_by_id()
    task_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    replay_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []

    counters: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    reward_counts: Counter[str] = Counter()
    worker_counts: Counter[str] = Counter()
    run_report_counts: Counter[str] = Counter()

    for report_path in sorted(manifest_dir.rglob("run_report.json")):
        counters["run_reports_seen"] += 1
        report = _read_json(report_path)
        job_map = _load_job_map(report, report_dir=report_path.parent)
        for row in report.get("rows") or []:
            counters["rows_seen"] += 1
            rollout_path = _resolve_path(row.get("rollout_out"), report_dir=report_path.parent)
            if not rollout_path or not rollout_path.exists():
                counters["missing_rollout"] += 1
                continue
            rollout = _read_json(rollout_path)
            job = job_map.get(str(row.get("job_id")))
            worker_names = _row_worker_names(row, job, rollout)
            if not any(is_commercial_worker(name) for name in worker_names):
                counters["non_commercial_rows"] += 1
                continue
            counters["commercial_rows"] += 1
            if not _is_complete_replay_record(row, rollout):
                counters["incomplete_commercial_rows"] += 1
                continue

            outcome = str(row.get("outcome_class") or rollout.get("outcome_class"))
            reward = rollout.get("reward")
            compact_workflow, allowed_workers = _compact_workflow(rollout, worker_names, by_id=by_id)
            task = _task_for_record(row, job, report_dir=report_path.parent, task_cache=task_cache)
            workflow_worker_names = _workflow_worker_names(rollout, by_id)
            run_report_rel = str(report_path.relative_to(REPO_ROOT)) if report_path.is_relative_to(REPO_ROOT) else str(report_path)

            replay_row = {
                "record_id": f"{report_path.parent.name}::{row.get('job_id')}",
                "task_id": row.get("task_id") or rollout.get("task_id"),
                "source_task_id": row.get("source_task_id"),
                "lane": row.get("lane"),
                "source": row.get("source") or rollout.get("source_name"),
                "task_split": row.get("task_split"),
                "arm": row.get("arm"),
                "stage": row.get("stage"),
                "worker_names": worker_names,
                "workflow_worker_names": workflow_worker_names,
                "commercial_worker_names": [name for name in worker_names if is_commercial_worker(name)],
                "outcome_class": outcome,
                "reward": reward,
                "grade": rollout.get("grade"),
                "valid_for_training": True,
                "conductor_sampled": bool((rollout.get("conductor") or {}).get("raw_output")),
                "run_report": run_report_rel,
                "rollout_json": str(rollout_path.relative_to(REPO_ROOT))
                if rollout_path.is_relative_to(REPO_ROOT)
                else str(rollout_path),
                "workflow": rollout.get("workflow"),
                "compact_workflow": compact_workflow,
                "execution": rollout.get("execution"),
            }
            replay_rows.append(replay_row)

            outcome_counts[outcome] += 1
            reward_counts[str(reward)] += 1
            for name in worker_names:
                worker_counts[name] += 1
            run_report_counts[run_report_rel] += 1

            if outcome == SUCCESS_OUTCOME and float(reward) == 1.0 and task is not None:
                sft_rows.append(
                    {
                        "record_id": replay_row["record_id"],
                        "task_id": replay_row["task_id"],
                        "lane": replay_row["lane"],
                        "source": replay_row["source"],
                        "arm": replay_row["arm"],
                        "allowed_workers": allowed_workers,
                        "workflow": compact_workflow,
                        "messages": [
                            *_sft_messages(task, allowed_workers),
                            {"role": "assistant", "content": json.dumps(compact_workflow, sort_keys=True)},
                        ],
                    }
                )
            elif outcome == SUCCESS_OUTCOME and task is None:
                counters["successful_records_missing_task"] += 1

    replay_path = out_dir / "commercial_rollout_replay.jsonl"
    sft_path = out_dir / "commercial_workflow_sft.jsonl"
    report_path = out_dir / "commercial_replay_report.json"
    _write_jsonl(replay_path, replay_rows)
    _write_jsonl(sft_path, sft_rows)

    report = {
        "manifest_dir": str(manifest_dir),
        "out_dir": str(out_dir),
        "replay_jsonl": str(replay_path),
        "sft_jsonl": str(sft_path),
        "counts": {
            **dict(counters),
            "complete_commercial_replay_records": len(replay_rows),
            "successful_workflow_sft_records": len(sft_rows),
            "conductor_sampled_records": sum(1 for row in replay_rows if row["conductor_sampled"]),
        },
        "outcome_counts": dict(outcome_counts),
        "reward_counts": dict(reward_counts),
        "worker_counts": dict(worker_counts),
        "run_report_counts": dict(run_report_counts),
        "use_policy": {
            "usable_for": [
                "workflow SFT warm-start",
                "offline replay analysis",
                "preference/ranking construction",
                "task and workflow prioritization before paid GRPO",
            ],
            "not_sufficient_for": [
                "standard online GRPO replacement without current-policy samples/logprobs",
                "final Ultra performance claim",
            ],
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    report = build_commercial_replay(Path(args.manifest_dir), Path(args.out_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
