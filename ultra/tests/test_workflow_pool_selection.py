import json

from ultra.workflow_pool_selection import (
    build_workflow_pool_selection_report,
    estimate_leave_one_out,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rollout(path, job_id, reward):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rollout_id": job_id,
                "reward": reward,
                "grade": {"score": reward, "success": reward >= 1.0},
                "valid_for_training": True,
                "outcome_class": "valid_correct_trainable" if reward >= 1.0 else "valid_incorrect_trainable",
            }
        )
        + "\n"
    )


def _job(job_id, task_id, arm, stage, workers):
    return {
        "job_id": job_id,
        "tournament_task_id": task_id,
        "lane": "repo_open_repo_terminal",
        "arm_domain": "terminal_sandbox",
        "source": "tasktrove_inferredbugs",
        "source_task_id": task_id,
        "arm": arm,
        "stage": stage,
        "worker_names": workers,
    }


def _write_shard(manifest_dir):
    jobs_jsonl = manifest_dir / "scaffold_tournament_jobs.jsonl"
    out_dir = manifest_dir / "scaffold_discovery_high_reasoning"
    jobs = [
        _job("job_1", "task_a", "solo__terminal_kimi_agent", "single_scaffold", ["terminal_kimi_agent"]),
        _job(
            "job_2",
            "task_a",
            "terminal_kimi_attempt__mimo_repair",
            "role_workflow",
            ["terminal_kimi_agent", "terminal_mimo_agent"],
        ),
        _job("job_3", "task_a", "solo__terminal_gpt_agent", "single_scaffold", ["terminal_gpt_agent"]),
        _job("job_4", "task_b", "solo__terminal_kimi_agent", "single_scaffold", ["terminal_kimi_agent"]),
        _job(
            "job_5",
            "task_b",
            "terminal_gpt_plan__kimi_solve",
            "role_workflow",
            ["terminal_gpt_agent", "terminal_kimi_agent"],
        ),
        _job("job_6", "task_b", "solo__terminal_mimo_agent", "single_scaffold", ["terminal_mimo_agent"]),
    ]
    _write_jsonl(jobs_jsonl, jobs)
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 0.5)
    _rollout(out_dir / "rollouts" / "job_2.json", "job_2", 1.0)
    _rollout(out_dir / "rollouts" / "job_3.json", "job_3", 0.5)
    _rollout(out_dir / "rollouts" / "job_4.json", "job_4", 1.0)
    _rollout(out_dir / "rollouts" / "job_5.json", "job_5", 1.0)
    _rollout(out_dir / "rollouts" / "job_6.json", "job_6", 0.5)
    return jobs_jsonl, out_dir


def test_estimate_leave_one_out_detects_role_worker_contribution(tmp_path):
    jobs_jsonl, out_dir = _write_shard(tmp_path)
    from ultra.conductor_baselines import load_rollout_rows

    stats = estimate_leave_one_out(load_rollout_rows(jobs_jsonl, out_dir))

    assert stats["terminal_mimo_agent"]["success_drop_groups"] == 1
    assert stats["terminal_mimo_agent"]["positive_reward_contribution_groups"] == 1
    assert stats["terminal_gpt_agent"]["success_drop_groups"] == 0
    assert stats["terminal_kimi_agent"]["success_drop_groups"] == 2


def test_build_workflow_pool_selection_report_writes_outputs(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _write_shard(manifest_dir)

    json_out = tmp_path / "pool.json"
    md_out = tmp_path / "pool.md"
    report = build_workflow_pool_selection_report(
        manifest_dir=manifest_dir,
        report_out=json_out,
        md_out=md_out,
    )

    assert report["version"] == "fugu_ultra_workflow_pool_selection_v1"
    assert report["task_summary"]["reward_variance_groups"] == 2
    assert "terminal_kimi_agent" in report["recommendations"]["recommended_mvp_grpo_workers"]
    assert json_out.exists()
    assert "Workflow Pool Selection" in md_out.read_text()
