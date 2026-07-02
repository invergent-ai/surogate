"""Build the Fugu-Ultra workflow-SFT warm-start dataset.

This stage is deliberately offline: it merges successful commercial-inclusive
workflow examples with deterministic topology priors from the registered fixed
workflow arms. It emits a conversation JSONL and a Surogate SFT config for the
Qwen3-8B Conductor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .commercial_replay import SYSTEM_PROMPT, _messages_text
from .scaffold_tournament import canonical_arms, canonical_workers

REPO_ROOT = Path(__file__).resolve().parents[2]

DOMAIN_TO_LANE = {
    "repo_coding": "trace_state_branches",
    "terminal_sandbox": "repo_open_repo_terminal",
    "unit_and_scientific_code": "unit_and_scientific_code",
    "math_science_knowledge": "math_science_knowledge",
    "tool_dialogue": "tool_dialogue",
    "long_context_memory_planning": "long_context_memory_planning",
}

DOMAIN_HARNESS_PREFERENCES = {
    "repo_coding": {"opencode", "opencode_repo", "codex", "claude_code"},
    "terminal_sandbox": {"terminal_sandbox"},
    "unit_and_scientific_code": {"code_exec"},
    "math_science_knowledge": {"direct_qa"},
    "tool_dialogue": {"tool_dialog", "tau_bench"},
    "long_context_memory_planning": {"long_context"},
}


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


def _worker_catalog_by_id() -> dict[int, dict[str, Any]]:
    return {int(worker.worker_id): worker.model_dump() for worker in canonical_workers()}


def _worker_catalog_by_name() -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in _worker_catalog_by_id().values()}


def _task_lane_map(pilot_config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lane, task_ids in (pilot_config.get("task_ids_by_lane") or {}).items():
        for task_id in task_ids:
            out[str(task_id)] = str(lane)
    return out


def _workers_for_names(worker_names: list[str]) -> list[dict[str, Any]]:
    by_name = _worker_catalog_by_name()
    workers = []
    for worker_id, name in enumerate(worker_names):
        worker = by_name.get(name, {})
        workers.append(
            {
                "worker_id": worker_id,
                "name": name,
                "backend": worker.get("backend"),
                "model": worker.get("model"),
                "role_prior": worker.get("role_prior") or [],
            }
        )
    return workers


def _prompt_messages(task: dict[str, Any], allowed_workers: list[dict[str, Any]]) -> list[dict[str, str]]:
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


def _compact_arm_workflow(arm: Any, worker_names: list[str]) -> dict[str, Any]:
    by_id = _worker_catalog_by_id()
    local_by_name = {name: i for i, name in enumerate(worker_names)}
    steps = []
    for step in arm.workflow.steps:
        worker = by_id[int(step.worker_id)]
        name = str(worker["name"])
        if name not in local_by_name:
            local_by_name[name] = len(local_by_name)
            worker_names.append(name)
        steps.append(
            {
                "worker_id": local_by_name[name],
                "subtask": step.subtask,
                "access": list(step.access),
                "budget": step.budget,
            }
        )
    return {"steps": steps}


def _message_hash(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _commercial_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        if not row.get("messages") or not row.get("workflow"):
            continue
        rows.append(
            {
                "source_kind": "commercial_success",
                "record_id": row.get("record_id"),
                "task_id": row.get("task_id"),
                "lane": row.get("lane"),
                "source": row.get("source"),
                "arm": row.get("arm"),
                "allowed_workers": row.get("allowed_workers") or [],
                "workflow": row.get("workflow"),
                "messages": row["messages"],
            }
        )
    return rows


def _eligible_tasks_by_domain(
    tasks: list[dict[str, Any]],
    pilot_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lane_by_task = _task_lane_map(pilot_config)
    for task in tasks:
        lane = lane_by_task.get(str(task.get("task_id")))
        if lane:
            by_lane[lane].append(task)

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain, lane in DOMAIN_TO_LANE.items():
        candidates = list(by_lane.get(lane, []))
        harnesses = DOMAIN_HARNESS_PREFERENCES.get(domain) or set()
        preferred = [task for task in candidates if (task.get("environment") or {}).get("harness") in harnesses]
        by_domain[domain] = sorted(preferred or candidates, key=lambda row: str(row.get("task_id")))
    return by_domain


def _topology_rows(
    tasks_jsonl: Path,
    pilot_config_json: Path,
    *,
    examples_per_arm: int,
) -> list[dict[str, Any]]:
    tasks = _read_jsonl(tasks_jsonl)
    pilot_config = _read_json(pilot_config_json)
    tasks_by_domain = _eligible_tasks_by_domain(tasks, pilot_config)
    rows: list[dict[str, Any]] = []
    for arm in canonical_arms():
        candidates = tasks_by_domain.get(arm.domain) or []
        if not candidates:
            continue
        for task in candidates[:examples_per_arm]:
            worker_names = list(arm.worker_names)
            workflow = _compact_arm_workflow(arm, worker_names)
            allowed_workers = _workers_for_names(worker_names)
            messages = [
                *_prompt_messages(task, allowed_workers),
                {"role": "assistant", "content": json.dumps(workflow, sort_keys=True)},
            ]
            rows.append(
                {
                    "source_kind": "topology_prior",
                    "record_id": f"topology::{arm.name}::{task.get('task_id')}",
                    "task_id": task.get("task_id"),
                    "lane": DOMAIN_TO_LANE.get(arm.domain),
                    "source": (task.get("source") or {}).get("name"),
                    "arm": arm.name,
                    "arm_domain": arm.domain,
                    "allowed_workers": allowed_workers,
                    "workflow": workflow,
                    "messages": messages,
                }
            )
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    duplicates = 0
    for row in rows:
        key = _message_hash(row["messages"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        out.append(row)
    return out, duplicates


def _write_sft_config(path: Path, dataset_path: Path, output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""model: "Qwen/Qwen3-8B"
output_dir: {output_dir}

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
sample_packing: true
sequence_len: 8192

max_steps: 150
eval_steps: 25
save_steps: 25
save_total_limit: 3
logging_steps: 1

learning_rate: 1e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
max_grad_norm: 1.0
weight_decay: 0.01
optimizer: adamw

recipe: bf16
recompute: true
cpu_training: true
merge_adapter: true
template: qwen3_nothinking
loss_scale: default

lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

dataloader_num_workers: 4
validation_split_ratio: 0.05
report_to:
  - surogate

datasets:
  - path: "{dataset_path}"
    type: conversation
    messages_field: messages
"""
    path.write_text(text, encoding="utf-8")


def build_workflow_sft_warmstart(
    *,
    commercial_sft_jsonl: Path,
    tasks_jsonl: Path,
    pilot_config_json: Path,
    out_dir: Path,
    examples_per_arm: int = 2,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    commercial = _commercial_rows(commercial_sft_jsonl)
    topology = _topology_rows(tasks_jsonl, pilot_config_json, examples_per_arm=examples_per_arm)
    rows, duplicate_count = _dedupe_rows([*commercial, *topology])

    dataset_path = out_dir / "workflow_sft_warmstart.jsonl"
    config_path = out_dir / "train_workflow_sft_qwen3_8b.yaml"
    report_path = out_dir / "workflow_sft_warmstart_report.json"
    output_dir = REPO_ROOT / "output" / "fugu_ultra_workflow_sft_qwen3_8b"

    _write_jsonl(dataset_path, rows)
    _write_sft_config(config_path, dataset_path.resolve(), output_dir)

    source_kind_counts = Counter(str(row.get("source_kind")) for row in rows)
    lane_counts = Counter(str(row.get("lane")) for row in rows)
    arm_counts = Counter(str(row.get("arm")) for row in rows)
    report = {
        "commercial_sft_jsonl": str(commercial_sft_jsonl),
        "tasks_jsonl": str(tasks_jsonl),
        "pilot_config_json": str(pilot_config_json),
        "out_dir": str(out_dir),
        "dataset_jsonl": str(dataset_path),
        "sft_config": str(config_path),
        "counts": {
            "commercial_input_rows": len(commercial),
            "topology_input_rows": len(topology),
            "dedupe_removed": duplicate_count,
            "total_rows": len(rows),
        },
        "source_kind_counts": dict(source_kind_counts),
        "lane_counts": dict(lane_counts),
        "top_arms": dict(arm_counts.most_common(25)),
        "policy": {
            "live_provider_calls": False,
            "purpose": "workflow-SFT warm start before capped commercial-inclusive GRPO",
            "not_a_performance_claim": True,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commercial-sft-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--pilot-config-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--examples-per-arm", type=int, default=2)
    args = parser.parse_args(argv)
    report = build_workflow_sft_warmstart(
        commercial_sft_jsonl=Path(args.commercial_sft_jsonl),
        tasks_jsonl=Path(args.tasks_jsonl),
        pilot_config_json=Path(args.pilot_config_json),
        out_dir=Path(args.out_dir),
        examples_per_arm=args.examples_per_arm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
