import json

from ultra.grpo_pilot_seed import build_grpo_pilot_seed


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _task(task_id):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "terminal_agentic",
        "source": {"name": "tasktrove_inferredbugs", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": f"Fix {task_id}"}]},
        "environment": {"harness": "terminal_sandbox"},
        "grader": {"type": "container_command", "command": ["pytest"], "expected_answer": None},
        "splitting": {"group_id": "g", "split": "grpo_train", "contamination_group": task_id},
        "metadata": {"domain": "terminal"},
    }


def _job(job_id, task_jsonl, task_id, arm, stage, reward):
    return {
        "job_id": job_id,
        "tournament_task_id": f"repo_open_repo_terminal::tasktrove_inferredbugs::{task_id}",
        "lane": "repo_open_repo_terminal",
        "arm_domain": "terminal_sandbox",
        "source": "tasktrove_inferredbugs",
        "source_task_id": task_id,
        "task_jsonl": str(task_jsonl),
        "task_harness": "terminal_sandbox",
        "arm": arm,
        "stage": stage,
        "worker_names": ["terminal_kimi_agent"],
        "_reward": reward,
    }


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


def test_build_grpo_pilot_seed_selects_only_disagreement_tasks(tmp_path):
    manifest_dir = tmp_path / "manifest"
    tasks_jsonl = manifest_dir / "tasks.jsonl"
    _write_jsonl(tasks_jsonl, [_task("task-a"), _task("task-b")])

    jobs = [
        _job("job_1", tasks_jsonl, "task-a", "solo__terminal_kimi_agent", "single_scaffold", 0.5),
        _job("job_2", tasks_jsonl, "task-a", "terminal_kimi_attempt__mimo_repair", "role_workflow", 1.0),
        _job("job_3", tasks_jsonl, "task-b", "solo__terminal_kimi_agent", "single_scaffold", 1.0),
        _job("job_4", tasks_jsonl, "task-b", "terminal_kimi_attempt__mimo_repair", "role_workflow", 1.0),
    ]
    _write_jsonl(
        manifest_dir / "scaffold_tournament_jobs.jsonl",
        [{k: v for k, v in job.items() if k != "_reward"} for job in jobs],
    )
    out_dir = manifest_dir / "scaffold_discovery_high_reasoning"
    for job in jobs:
        _rollout(out_dir / "rollouts" / f"{job['job_id']}.json", job["job_id"], job["_reward"])

    seed_out = tmp_path / "pilot" / "seed.jsonl"
    tasks_out = tmp_path / "pilot" / "tasks.jsonl"
    report = build_grpo_pilot_seed(
        manifest_dir=manifest_dir,
        out_jsonl=seed_out,
        report_out=tmp_path / "pilot" / "report.json",
        task_jsonl_out=tasks_out,
    )

    selected = [json.loads(line) for line in seed_out.read_text().splitlines()]
    materialized = [json.loads(line) for line in tasks_out.read_text().splitlines()]
    assert report["selected_tasks"] == 1
    assert report["materialized_tasks"] == 1
    assert selected[0]["source_task_id"] == "task-a"
    assert selected[0]["selection_reasons"] == [
        "reward_variance",
        "role_beats_best_single",
        "workflow_oracle_headroom",
    ]
    assert materialized[0]["task_id"] == "task-a"


def test_build_grpo_pilot_seed_merges_same_tournament_task_across_task_jsonls(tmp_path):
    manifest_dir = tmp_path / "manifest"
    tasks_jsonl_a = manifest_dir / "source_a" / "tasks.jsonl"
    tasks_jsonl_b = manifest_dir / "source_b" / "tasks.jsonl"
    _write_jsonl(tasks_jsonl_a, [_task("task-a")])
    _write_jsonl(tasks_jsonl_b, [_task("task-a")])

    jobs = [
        _job("job_1", tasks_jsonl_a, "task-a", "solo__terminal_kimi_agent", "single_scaffold", 1.0),
        _job("job_2", tasks_jsonl_b, "task-a", "solo__terminal_mimo_agent", "single_scaffold", 0.5),
    ]
    _write_jsonl(
        manifest_dir / "scaffold_tournament_jobs.jsonl",
        [{k: v for k, v in job.items() if k != "_reward"} for job in jobs],
    )
    out_dir = manifest_dir / "scaffold_discovery_high_reasoning"
    for job in jobs:
        _rollout(out_dir / "rollouts" / f"{job['job_id']}.json", job["job_id"], job["_reward"])

    seed_out = tmp_path / "pilot" / "seed.jsonl"
    tasks_out = tmp_path / "pilot" / "tasks.jsonl"
    report = build_grpo_pilot_seed(
        manifest_dir=manifest_dir,
        out_jsonl=seed_out,
        report_out=tmp_path / "pilot" / "report.json",
        task_jsonl_out=tasks_out,
    )

    selected = [json.loads(line) for line in seed_out.read_text().splitlines()]
    assert report["selected_tasks"] == 1
    assert report["materialized_tasks"] == 1
    assert selected[0]["source_task_id"] == "task-a"
    assert selected[0]["selection_reasons"] == ["reward_variance"]
    assert selected[0]["reward_values"] == [0.5, 1.0]


def test_build_grpo_pilot_seed_resolves_repo_relative_task_jsonl(tmp_path):
    repo = tmp_path / "repo"
    manifest_dir = repo / "director" / "manifests" / "fugu_clean_v1"
    tasks_jsonl = manifest_dir / "source" / "tasks.jsonl"
    _write_jsonl(tasks_jsonl, [_task("task-a")])
    bad_relative = "../director/manifests/fugu_clean_v1/source/tasks.jsonl"
    jobs = [
        _job("job_1", bad_relative, "task-a", "solo__terminal_kimi_agent", "single_scaffold", 0.5),
        _job("job_2", bad_relative, "task-a", "terminal_kimi_attempt__mimo_repair", "role_workflow", 1.0),
    ]
    _write_jsonl(
        manifest_dir / "scaffold_tournament_jobs.jsonl",
        [{k: v for k, v in job.items() if k != "_reward"} for job in jobs],
    )
    out_dir = manifest_dir / "scaffold_discovery_high_reasoning"
    for job in jobs:
        _rollout(out_dir / "rollouts" / f"{job['job_id']}.json", job["job_id"], job["_reward"])

    tasks_out = tmp_path / "pilot" / "tasks.jsonl"
    report = build_grpo_pilot_seed(
        manifest_dir=manifest_dir,
        out_jsonl=tmp_path / "pilot" / "seed.jsonl",
        report_out=tmp_path / "pilot" / "report.json",
        task_jsonl_out=tasks_out,
    )

    assert report["materialized_tasks"] == 1
    assert json.loads(tasks_out.read_text())["task_id"] == "task-a"
