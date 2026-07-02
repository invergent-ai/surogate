import json

from ultra.manifest_freeze import build_manifest_freeze


def _task(task_id, source, split, harness="direct_qa", domain="math"):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "math",
        "source": {"name": source, "version": "v1", "policy": "final_eval_only" if split == "final_eval" else "train_allowed"},
        "input": {"messages": [{"role": "user", "content": f"Task {task_id}"}]},
        "environment": {"harness": harness},
        "grader": {"type": "exact_match", "expected_answer": "ok"},
        "splitting": {
            "group_id": source,
            "split": split,
            "contamination_group": f"{source}::{task_id}",
        },
        "metadata": {"domain": domain},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_manifest_freeze_writes_hash_locked_manifests(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _write_jsonl(
        manifest_dir / "data_mix" / "existing_bank_taskspecs.jsonl",
        [_task("online-1", "existing_bank", "online_validation")],
    )
    _write_jsonl(
        manifest_dir / "scaffold_repo_taskspecs.jsonl",
        [_task("deep-1", "deep_swe_local", "final_eval", harness="opencode", domain="software_engineering")],
    )
    _write_jsonl(
        manifest_dir / "pool_matrix_frontier.jsonl",
        [{"task_id": "pool-1", "domain": "math", "worker_ids": ["gpt"], "rewards": [1]}],
    )
    _write_jsonl(
        manifest_dir / "agentic_frontier_tau4.jsonl",
        [{"item_id": "tau-1", "domain": "tau_retail", "worker": "opus", "reward": 1}],
    )
    _write_jsonl(
        manifest_dir / "agentic_coding_frontier_direct3.jsonl",
        [{"task_id": "code-1", "arm": "opencode:kimi", "reward": 1}],
    )

    report = build_manifest_freeze(
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "frozen",
        report_out=manifest_dir / "frozen" / "freeze_report.json",
        md_out=manifest_dir / "frozen" / "freeze_report.md",
        created_at_utc="2026-06-27T00:00:00Z",
    )

    by_name = {item["manifest_name"]: item for item in report["manifests"]}
    assert report["freeze_complete"] is True
    assert by_name["online_validation"]["row_count"] == 1
    assert by_name["pool_validation"]["row_count"] == 3
    assert by_name["final_eval"]["row_count"] == 1
    assert by_name["deep_swe_target_eval"]["source_counts"] == {"deep_swe_local": 1}
    assert by_name["online_validation"]["sha256"].startswith("sha256:")
    assert report["checks"]["online_final_overlap_count"] == 0
    assert (manifest_dir / "frozen" / "freeze_report.md").exists()
