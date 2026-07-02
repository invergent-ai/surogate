import json

from ultra.grpo_pilot_freeze import build_grpo_pilot_freeze


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _task(task_id, source="existing_bank", lane="math_science_knowledge", contamination=None):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "math",
        "source": {"name": source, "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": f"Task {task_id}"}]},
        "environment": {"harness": "direct_qa"},
        "grader": {"type": "exact_match", "expected_answer": "ok"},
        "splitting": {
            "group_id": source,
            "split": "grpo_train",
            "contamination_group": contamination or f"{source}::{task_id}",
        },
        "metadata": {"domain": "math", "tags": [lane]},
    }


def _seed(task_id, lane="math_science_knowledge", source="existing_bank"):
    return {
        "pilot_task_id": f"{lane}::{source}::{task_id}",
        "tournament_task_id": f"{lane}::{source}::{task_id}",
        "lane": lane,
        "source": source,
        "source_task_id": task_id,
        "task_jsonl": "tasks.jsonl",
        "task_harness": "direct_qa",
        "selection_reasons": ["reward_variance"],
        "reward_values": [0.5, 1.0],
        "rollouts_observed": 2,
        "recommended_group_size": 8,
    }


def _eval_task(task_id, split="final_eval", source="deep_swe_local", contamination=None):
    row = _task(task_id, source=source, contamination=contamination)
    row["source"]["policy"] = "final_eval_only"
    row["splitting"]["split"] = split
    return row


def test_grpo_pilot_freeze_writes_hash_locked_training_manifest(tmp_path):
    manifest_dir = tmp_path / "manifest"
    tasks = [_task("task-a"), _task("task-b", lane="repo_open_repo_terminal")]
    seeds = [_seed("task-a"), _seed("task-b", lane="repo_open_repo_terminal")]
    tasks_jsonl = manifest_dir / "grpo_pilot_seed" / "taskspecs.jsonl"
    seed_jsonl = manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl"
    _write_jsonl(tasks_jsonl, tasks)
    _write_jsonl(seed_jsonl, seeds)
    _write_jsonl(manifest_dir / "frozen_manifests" / "online_validation.jsonl", [])
    _write_jsonl(manifest_dir / "frozen_manifests" / "final_eval.jsonl", [_eval_task("eval-1")])
    _write_jsonl(manifest_dir / "frozen_manifests" / "deep_swe_target_eval.jsonl", [_eval_task("deep-1")])
    (manifest_dir / "grpo_pilot_seed" / "gap_plan.json").write_text(
        json.dumps({"lane_deficits": {"math_science_knowledge": 0}, "target_lane_counts": {"math_science_knowledge": 1}})
    )

    report = build_grpo_pilot_freeze(
        manifest_dir=manifest_dir,
        seed_jsonl=seed_jsonl,
        tasks_jsonl=tasks_jsonl,
        out_dir=manifest_dir / "grpo_pilot_train",
        report_out=manifest_dir / "grpo_pilot_train" / "freeze_report.json",
        md_out=manifest_dir / "grpo_pilot_train" / "freeze_report.md",
        target_task_count=2,
        created_at_utc="2026-06-28T00:00:00Z",
    )

    assert report["freeze_complete"] is True
    assert report["task_count"] == 2
    assert report["checks"]["final_eval_task_id_overlap"] == 0
    assert report["manifests"][0]["sha256"].startswith("sha256:")
    assert (manifest_dir / "grpo_pilot_train" / "taskspecs.jsonl").exists()
    assert (manifest_dir / "grpo_pilot_train" / "freeze_report.md").exists()


def test_grpo_pilot_freeze_rejects_eval_contamination_overlap(tmp_path):
    manifest_dir = tmp_path / "manifest"
    train = _task("task-a", contamination="shared-group")
    tasks_jsonl = manifest_dir / "grpo_pilot_seed" / "taskspecs.jsonl"
    seed_jsonl = manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl"
    _write_jsonl(tasks_jsonl, [train])
    _write_jsonl(seed_jsonl, [_seed("task-a")])
    _write_jsonl(manifest_dir / "frozen_manifests" / "online_validation.jsonl", [])
    _write_jsonl(
        manifest_dir / "frozen_manifests" / "final_eval.jsonl",
        [_eval_task("eval-1", contamination="shared-group")],
    )
    _write_jsonl(manifest_dir / "frozen_manifests" / "deep_swe_target_eval.jsonl", [])

    report = build_grpo_pilot_freeze(
        manifest_dir=manifest_dir,
        seed_jsonl=seed_jsonl,
        tasks_jsonl=tasks_jsonl,
        out_dir=manifest_dir / "grpo_pilot_train",
        target_task_count=1,
    )

    assert report["freeze_complete"] is False
    assert report["checks"]["eval_contamination_group_overlap"] == 1
