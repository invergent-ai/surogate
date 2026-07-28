"""CRMArena training manifests (office lane 5): offline CRM query tasks.

The benchmark's org data is published as local SQLite (no Salesforce org
needed); tasks come from crmarena_w_metadata.json. Only the 1,040
EXACT-MATCH tasks are used — the fuzzy-match type (knowledge_qa) needs an
LLM judge, which is out of scope for training rewards.

CRMArena publishes NO train split, so training on it makes the published
number unreportable for us; instead a stratified 20% holdout (26 of 130 per
task type, sha1-ordered) is SEALED for internal eval and the remaining 80%
trains. Both files carry the split explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CAPABILITY = "office_crm"
HOLDOUT_FRACTION = 0.2
DEFAULT_MAX_TURNS = 16

PROMPT_NOTE = ("Answer the question by querying the CRM database. Reply with "
               "ONLY the answer (an Id, a name, or a number); reply None if "
               "there is no valid answer.")


def _context(meta: dict[str, Any]) -> str:
    parts = []
    for key in ("required", "optional"):
        text = str(meta.get(key) or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def build_taskspec(row: dict[str, Any], db_path: Path, *, split: str,
                   max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, Any]:
    idx = int(row["idx"])
    task_type = str(row["task"])
    return {
        "schema_version": "2.0",
        "task_id": f"crmarena__{task_type}__{idx:04d}",
        "capability": CAPABILITY,
        "source": {
            "name": "crmarena", "url_or_ref": "https://huggingface.co/datasets/Salesforce/CRMArena",
            "version": "crmarena-v1", "license": "CC BY-NC 4.0",
            "policy": "train_allowed" if split == "grpo_train" else "final_eval_only",
            "source_commit": None,
        },
        "input": {"messages": [
            {"role": "user", "content": f"{PROMPT_NOTE}\n\n{str(row['query']).strip()}"}],
            "context_documents": [], "assets": [], "repo": None, "tools": []},
        "environment": {"harness": "crm_query", "image": None, "cpu_limit": None,
                        "memory_mb": None, "disk_mb": None,
                        "network_policy": "model-relay-only", "wall_time_seconds": 900},
        "grader": {"type": "crm_exact", "command": None,
                   "expected_answer": {
                       "db_path": str(db_path.resolve()),
                       "answer": row.get("answer"),
                       "context": _context(row.get("metadata") or {}),
                       "max_turns": max_turns,
                   },
                   "score_range": [0.0, 1.0], "success_threshold": 1.0,
                   "deterministic": True},
        "splitting": {"split": split, "group_id": f"crmarena_{task_type}",
                      "contamination_group": f"crmarena/{task_type}/{idx}"},
        "metadata": {"domain": CAPABILITY, "subdomain": task_type,
                     "difficulty_estimate": None,
                     "estimated_worker_calls": max_turns, "requires_tools": True,
                     "requires_long_context": False,
                     "tags": ["crmarena", "crm", task_type, split, "exact_match"]},
    }


def export(tasks_json: Path, db_path: Path, out_dir: Path,
           *, holdout_fraction: float = HOLDOUT_FRACTION) -> dict[str, Any]:
    rows = json.loads(tasks_json.read_text())
    exact = [r for r in rows if r.get("reward_metric") == "exact_match"]
    excluded_fuzzy = len(rows) - len(exact)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact:
        by_type[str(row["task"])].append(row)

    train_specs: list[str] = []
    holdout_specs: list[str] = []
    for task_type, group in sorted(by_type.items()):
        ordered = sorted(group, key=lambda r: hashlib.sha1(
            f"{task_type}/{r['idx']}".encode()).hexdigest())
        n_holdout = max(1, round(len(ordered) * holdout_fraction))
        for i, row in enumerate(ordered):
            split = "final_eval" if i < n_holdout else "grpo_train"
            spec = json.dumps(build_taskspec(row, db_path, split=split), sort_keys=True)
            (holdout_specs if i < n_holdout else train_specs).append(spec)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "crmarena_train_taskspecs.jsonl"
    holdout_path = out_dir / "crmarena_sealed_holdout_taskspecs.jsonl"
    train_path.write_text("".join(s + "\n" for s in train_specs))
    holdout_path.write_text("".join(s + "\n" for s in holdout_specs))
    return {
        "exact_match_tasks": len(exact),
        "excluded_fuzzy": excluded_fuzzy,
        "train": len(train_specs),
        "sealed_holdout": len(holdout_specs),
        "task_types": len(by_type),
        "train_manifest": str(train_path),
        "holdout_manifest": str(holdout_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.tasks, args.db, args.out_dir), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
