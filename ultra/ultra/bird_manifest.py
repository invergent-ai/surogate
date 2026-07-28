"""BIRD text-to-SQL training-manifest exporter (office lane 2).

BIRD ships a train split (train.json + train_databases) and a dev split
(dev.json + dev_databases). DEV IS THE REPORTABLE BENCHMARK and is sealed:
``export`` refuses to write a training manifest from it.

Each task becomes a SINGLE-TURN row graded by ``sql_exec`` (execution
accuracy against the task's own SQLite file, read-only, subprocess-isolated).
The prompt carries the question, BIRD's ``evidence`` hint, and the database
DDL — the worker answers with one SQLite query in a ``sql`` fence.

DDL is read from the database itself rather than BIRD's tables.json so the
schema shown to the worker is exactly what the grader will execute against.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from ultra.grading.verifiers import _sql_digest

CAPABILITY = "office_data_query"
TRAIN_SPLIT = "train"
SEALED_SPLIT = "dev"  # published BIRD benchmark: eval only

# TaskMetadata.difficulty_estimate is numeric; BIRD ships a categorical label,
# which is kept verbatim in tags.
DIFFICULTY_ESTIMATE = {"simple": 0.25, "moderate": 0.55, "challenging": 0.85}

PROMPT_TEMPLATE = """\
Answer the question by writing ONE SQLite query against the database below.

Database schema:
```sql
{schema}
```

{evidence}Question: {question}

Reply with the query in a ```sql fenced block. Return exactly the columns the \
question asks for."""


def _schema_ddl(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(r[0].strip() for r in rows)


def _db_path(dataset_dir: Path, split: str, db_id: str) -> Path:
    return (dataset_dir / f"{split}_databases" / db_id / f"{db_id}.sqlite").resolve()


def build_taskspec(row: dict[str, Any], dataset_dir: Path, split: str,
                   *, schema: str, timeout: float = 30.0,
                   index: int | None = None) -> dict[str, Any]:
    # BIRD's splits differ: dev rows carry question_id and difficulty, train
    # rows carry neither. Fall back to the row's position, which is stable
    # for a given release of train.json.
    db_id = str(row["db_id"])
    question_id = int(row.get("question_id", index if index is not None else 0))
    task_id = f"bird__{split}__{db_id}__{question_id:06d}"
    evidence = str(row.get("evidence") or "").strip()
    difficulty = str(row.get("difficulty") or "").strip()
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": CAPABILITY,
        "source": {
            "name": f"bird_{split}",
            "url_or_ref": "https://bird-bench.github.io/",
            "version": "bird-v1",
            "license": "CC BY-SA 4.0",
            "policy": "train_allowed",
            "source_commit": None,
        },
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        schema=schema,
                        evidence=f"Hint: {evidence}\n\n" if evidence else "",
                        question=str(row["question"]).strip(),
                    ),
                }
            ],
            "context_documents": [],
            "assets": [],
            "repo": None,
            "tools": [],
        },
        "environment": {
            "harness": "direct_qa",
            "image": None,
            "cpu_limit": None,
            "memory_mb": None,
            "disk_mb": None,
            "network_policy": "model-relay-only",
            "wall_time_seconds": 600,
        },
        "grader": {
            "type": "sql_exec",
            "command": None,
            "expected_answer": {
                "db_path": str(_db_path(dataset_dir, split, db_id)),
                "gold_sql": str(row["SQL"]).strip(),
                "timeout": timeout,
            },
            "score_range": [0.0, 1.0],
            "success_threshold": 1.0,
            "deterministic": True,
        },
        "splitting": {
            "split": "grpo_train",
            "group_id": f"bird_{split}_{db_id}",
            "contamination_group": f"bird/{split}/{db_id}/{question_id}",
        },
        "metadata": {
            "domain": CAPABILITY,
            "subdomain": "bird_text_to_sql",
            "difficulty_estimate": DIFFICULTY_ESTIMATE.get(difficulty),
            "estimated_worker_calls": 1,
            "requires_tools": False,
            "requires_long_context": False,
            "tags": ["bird", "text_to_sql", split, "execution_accuracy"]
                    + ([f"difficulty:{difficulty}"] if difficulty else []),
        },
    }


def export(dataset_dir: Path, out_dir: Path, *, split: str = TRAIN_SPLIT,
           timeout: float = 30.0, limit: int | None = None) -> dict[str, Any]:
    """Write bird_<split>_taskspecs.jsonl; refuses to train on the sealed split."""
    if split == SEALED_SPLIT:
        raise RuntimeError(
            f"BIRD {SEALED_SPLIT!r} is the reportable benchmark and is sealed for "
            "evaluation — refusing to build a training manifest from it"
        )
    questions = json.loads((dataset_dir / f"{split}.json").read_text())
    if limit is not None:
        questions = questions[:limit]

    schemas: dict[str, str] = {}
    rows: list[str] = []
    missing_dbs: set[str] = set()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    for index, row in enumerate(questions):
        db_id = str(row["db_id"])
        if db_id in missing_dbs:
            continue
        if db_id not in schemas:
            db = _db_path(dataset_dir, split, db_id)
            if not db.exists():
                missing_dbs.add(db_id)
                continue
            schemas[db_id] = _schema_ddl(db)
        spec = build_taskspec(row, dataset_dir, split, schema=schemas[db_id],
                              timeout=timeout, index=index)
        # Task ids key the probe journal and the lane config, so a collision
        # would silently merge two different questions.
        if spec["task_id"] in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(spec["task_id"])
        rows.append(json.dumps(spec, sort_keys=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"bird_{split}_taskspecs.jsonl"
    manifest.write_text("".join(line + "\n" for line in rows))
    return {
        "split": split,
        "questions": len(questions),
        "tasks": len(rows),
        "databases": len(schemas),
        "missing_databases": sorted(missing_dbs),
        "duplicate_task_ids": duplicate_ids,
        "manifest": str(manifest),
    }


LANE = "office_data_query"
DEFAULT_LANE_SIZE = 400
DEFAULT_GROUP_SIZE = 8
# Filename the campaign's MIX_FILES expects for BIRD probe candidates.
PROBE_SAMPLE_NAME = "bird_probe_candidates_taskspecs.jsonl"
DEFAULT_PROBE_SAMPLE = 400
# The conductor's prompt budget is sequence_len minus the plan cap (~5120
# tokens). Two of BIRD's 69 train databases carry schemas so large that the
# prompt alone runs ~9k tokens; capping the prompt at 12k chars (~3k tokens)
# keeps 92% of the questions and 67 of the databases while staying safely
# inside that budget.
MAX_PROMPT_CHARS = 12000


def write_probe_sample(
    manifest: Path,
    out_dir: Path,
    *,
    size: int = DEFAULT_PROBE_SAMPLE,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
    verify_gold: bool = True,
) -> dict[str, Any]:
    """Write the BIRD subset the probe should curate.

    BIRD's train split is far larger than a campaign needs, and probing a
    question costs 3 rollouts against the ~16 a training group spends — so
    the probe sees a deterministic sha1-ordered sample rather than all of
    it. Sampling by hash keeps the draw representative across databases and
    difficulty without selecting on content.

    Two upstream defects are filtered here rather than paid for later:
    oversized schemas that would blow the conductor's prompt budget, and
    train rows whose own gold SQL does not execute (~5% by sampling), which
    would score every attempt 0 and look like an impossibly hard question.
    Both counts are reported — nothing is dropped silently.
    """
    import hashlib

    rows = [line for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty BIRD manifest — nothing to probe")
    ordered = sorted(rows, key=lambda line: hashlib.sha1(
        json.loads(line)["task_id"].encode()).hexdigest())

    sample: list[str] = []
    oversized = broken_gold = 0
    for line in ordered:
        if len(sample) >= size:
            break
        spec = json.loads(line)
        if len(spec["input"]["messages"][0]["content"]) > max_prompt_chars:
            oversized += 1
            continue
        if verify_gold:
            payload = spec["grader"]["expected_answer"]
            if _sql_digest(payload["db_path"], payload["gold_sql"],
                           float(payload.get("timeout", 30.0))) is None:
                broken_gold += 1
                continue
        sample.append(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / PROBE_SAMPLE_NAME
    out_path.write_text("".join(line + "\n" for line in sample))
    databases = Counter(json.loads(line)["splitting"]["group_id"] for line in sample)
    return {
        "manifest_rows": len(rows),
        "probe_candidates": len(sample),
        "skipped_oversized_prompt": oversized,
        "skipped_broken_gold": broken_gold,
        "databases": len(databases),
        "path": str(out_path),
    }


def export_lane(
    manifest: Path,
    base_pilot_config: Path,
    binding_path: Path,
    out_dir: Path,
    *,
    lane_size: int = DEFAULT_LANE_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    approved: bool = False,
) -> dict[str, Any]:
    """Bounce-in artifacts for the BIRD lane: pilot config + safety manifest.

    Single-turn lane, so the worker pool is the CURRENT open-weight binding
    (same construction the probe-curated lanes use), not a tool-calling mask.
    ``approved`` defaults to False: approval is the operator's launch gate
    and must be passed explicitly, and carrying it as a parameter means
    regenerating the manifest preserves the decision.
    """
    import hashlib

    from ultra.live_worker_safety import VERSION as SAFETY_VERSION
    from ultra.probe_manifest_export import build_worker_pool

    ids = [json.loads(line)["task_id"] for line in manifest.read_text().splitlines()
           if line.strip()]
    if not ids:
        raise ValueError("empty BIRD manifest — refusing to build an empty lane")
    ordered = sorted(ids, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    task_ids = sorted(ordered[:lane_size])

    pool = build_worker_pool(json.loads(binding_path.read_text()))
    config = json.loads(base_pilot_config.read_text())
    config["version"] = "fugu_ultra_office_bird_r5"
    config["conductor_contract"] = "typed_control"
    config["worker_pool"] = pool
    config["worker_pool_names"] = sorted(pool)
    config["lane_worker_masks"] = {LANE: sorted(pool)}
    config["task_ids_by_lane"] = {LANE: task_ids}
    config["lane_counts"] = {LANE: len(task_ids)}
    config["task_count"] = len(task_ids)
    config["group_size_by_lane"] = {LANE: int(group_size)}
    config["provider_policy"] = {
        "gpt_never_openrouter": False,
        "openrouter_workers": sorted(pool),
        "yunwu_only_workers": [],
    }

    safety = {
        "version": SAFETY_VERSION,
        "approved": bool(approved),
        "purpose": (
            "r5 bounce-in: BIRD text-to-SQL lane (office_data_query). Single "
            "turn, open-weight workers, openrouter only, execution-accuracy "
            "reward against read-only SQLite. TRAIN SPLIT ONLY — BIRD dev is "
            "the reportable benchmark and is sealed."
        ),
        "allowed_lanes": [LANE],
        "max_workflow_steps": 5,
        "required_force_step_budget": "short",
        "max_examples_by_lane": {LANE: len(task_ids)},
        "allowed_providers": ["openrouter"],
        "allowed_workers_by_lane": {LANE: sorted(pool)},
        "allowed_yunwu_workers": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "pilot_config_r5_bird.json"
    safety_path = out_dir / "live_safety_r5_bird.json"
    config_path.write_text(json.dumps(config, indent=1))
    safety_path.write_text(json.dumps(safety, indent=1))
    return {
        "lane": LANE,
        "lane_tasks": len(task_ids),
        "pool": sorted(pool),
        "pilot_config": str(config_path),
        "live_safety": str(safety_path),
        "approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        help="directory holding <split>.json and <split>_databases/")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default=TRAIN_SPLIT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lane-from-manifest", type=Path, default=None,
                        help="build lane artifacts from this manifest instead of exporting")
    parser.add_argument("--probe-sample-from", type=Path, default=None,
                        help="write the probe-candidate subset from this manifest")
    parser.add_argument("--probe-sample-size", type=int, default=DEFAULT_PROBE_SAMPLE)
    parser.add_argument("--base-pilot-config", type=Path, default=None)
    parser.add_argument("--binding", type=Path, default=None)
    parser.add_argument("--lane-size", type=int, default=DEFAULT_LANE_SIZE)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--approved", action="store_true",
                        help="mark the safety manifest approved for live spend "
                             "(operator decision; default is unapproved)")
    args = parser.parse_args(argv)

    if args.probe_sample_from is not None:
        print(json.dumps(write_probe_sample(
            args.probe_sample_from, args.out_dir, size=args.probe_sample_size,
        ), indent=1))
        return 0

    if args.lane_from_manifest is not None:
        if args.base_pilot_config is None or args.binding is None:
            parser.error("--lane-from-manifest needs --base-pilot-config and --binding")
        print(json.dumps(export_lane(
            args.lane_from_manifest, args.base_pilot_config, args.binding,
            args.out_dir, lane_size=args.lane_size, group_size=args.group_size,
            approved=args.approved,
        ), indent=1))
        return 0

    if args.dataset_dir is None:
        parser.error("--dataset-dir is required unless --lane-from-manifest is given")
    print(json.dumps(export(args.dataset_dir, args.out_dir, split=args.split,
                            timeout=args.timeout, limit=args.limit), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
