import json

from ultra.discovery_followup import build_discovery_followup_plan


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


def _job(job_id, task_id, arm, stage):
    return {
        "job_id": job_id,
        "tournament_task_id": f"repo_open_repo_terminal::tasktrove_inferredbugs::{task_id}",
        "lane": "repo_open_repo_terminal",
        "arm_domain": "terminal_sandbox",
        "source": "tasktrove_inferredbugs",
        "source_task_id": task_id,
        "task_jsonl": "/tmp/tasks.jsonl",
        "task_harness": "terminal_sandbox",
        "arm": arm,
        "stage": stage,
        "worker_names": ["terminal_kimi_agent"],
    }


def test_followup_plans_missing_jobs_for_reward_variance_tasks(tmp_path):
    manifest_dir = tmp_path / "manifest"
    jobs_jsonl = manifest_dir / "scaffold_tournament_jobs.jsonl"
    out_dir = manifest_dir / "scaffold_discovery_high_reasoning"
    jobs = [
        _job("job_1", "task-a", "solo__terminal_kimi_agent", "single_scaffold"),
        _job("job_2", "task-a", "solo__terminal_mimo_agent", "single_scaffold"),
        _job("job_3", "task-a", "terminal_kimi_attempt__mimo_repair", "role_workflow"),
        _job("job_4", "task-b", "solo__terminal_kimi_agent", "single_scaffold"),
        _job("job_5", "task-b", "solo__terminal_mimo_agent", "single_scaffold"),
        _job("job_6", "task-b", "terminal_kimi_attempt__mimo_repair", "role_workflow"),
    ]
    _write_jsonl(jobs_jsonl, jobs)
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 0.5)
    _rollout(out_dir / "rollouts" / "job_2.json", "job_2", 1.0)
    _rollout(out_dir / "rollouts" / "job_4.json", "job_4", 1.0)
    _rollout(out_dir / "rollouts" / "job_5.json", "job_5", 1.0)

    report = build_discovery_followup_plan(
        manifest_dir=manifest_dir,
        out_json=tmp_path / "followup.json",
        jobs_out=tmp_path / "followup_jobs.jsonl",
        sources={"tasktrove_inferredbugs"},
        mode="targeted",
        max_jobs=8,
        max_task_groups=4,
    )

    assert report["selected_jobs"] == 1
    assert report["selected"][0]["job_id"] == "job_3"
    assert report["selected"][0]["reason"] == ["complete_reward_variance_task"]
    assert report["reason_counts"]["complete_reward_variance_task"] == 1
    assert "job_3" in (tmp_path / "followup_jobs.jsonl").read_text()


def test_single_prefilter_selects_fresh_single_jobs_only(tmp_path):
    manifest_dir = tmp_path / "manifest"
    jobs_jsonl = manifest_dir / "scaffold_tournament_jobs.jsonl"
    jobs = [
        _job("job_1", "task-a", "solo__terminal_kimi_agent", "single_scaffold"),
        _job("job_2", "task-a", "terminal_kimi_attempt__mimo_repair", "role_workflow"),
    ]
    _write_jsonl(jobs_jsonl, jobs)

    report = build_discovery_followup_plan(
        manifest_dir=manifest_dir,
        out_json=tmp_path / "followup.json",
        stages={"single_scaffold", "role_workflow"},
        mode="single-prefilter",
        max_jobs=8,
        max_task_groups=4,
    )

    assert report["selected_jobs"] == 1
    assert report["selected"][0]["job_id"] == "job_1"
    assert report["stage_counts"] == {"single_scaffold": 1}
