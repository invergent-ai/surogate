import json

import pytest

from ultra.trace_capture_plan import build_trace_capture_plan


def _repo_task(task_id, source="generated_repo_tasks", split="grpo_train", policy="train_allowed"):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "agentic_coding",
        "source": {"name": source, "version": "v1", "policy": policy},
        "input": {
            "messages": [{"role": "user", "content": "Fix the bug"}],
            "assets": [
                {
                    "opencode_instance": {
                        "image_name": "example/task:latest",
                        "problem_statement": "Fix the bug",
                    }
                }
            ],
        },
        "environment": {"harness": "opencode"},
        "grader": {"type": "deep_swe_hidden_tests"},
        "splitting": {
            "group_id": source,
            "split": split,
            "contamination_group": f"{source}::{task_id}",
        },
        "metadata": {"domain": "software_engineering"},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_trace_capture_plan_selects_train_allowed_generated_repo_jobs(tmp_path):
    manifest_dir = tmp_path / "manifest"
    tasks_jsonl = manifest_dir / "generated_repo_tasks" / "taskspecs.jsonl"
    rows = [_repo_task(f"generated_repo_tasks__t{i}") for i in range(3)]
    rows.append(_repo_task("training_repo_canary__x", source="training_repo_canary", split="diagnostic"))
    rows.append(_repo_task("deep_swe_local__x", source="deep_swe_local", split="final_eval", policy="final_eval_only"))
    _write_jsonl(tasks_jsonl, rows)

    plan = build_trace_capture_plan(
        manifest_dir=manifest_dir,
        tasks_jsonl=tasks_jsonl,
        out_json=manifest_dir / "trace_capture" / "plan.json",
        jobs_out=manifest_dir / "trace_capture" / "jobs.jsonl",
        task_limit=2,
        seed=0,
    )

    jobs = [
        json.loads(line)
        for line in (manifest_dir / "trace_capture" / "jobs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert plan["selected_task_count"] == 2
    assert plan["job_count"] == 6
    assert len(jobs) == 6
    assert {job["source_name"] for job in jobs} == {"generated_repo_tasks"}
    assert {job["split"] for job in jobs} == {"grpo_train"}
    assert {job["arm"] for job in jobs} == {
        "solo__opencode_kimi_builder",
        "solo__codex_gpt_coding_agent",
        "solo__claude_code_opus_debugger",
    }
    assert all("final_patch_ref" in job["required_artifacts"] for job in jobs)
    assert all(job["acceptance_gate"]["must_have_execution_feedback"] for job in jobs)


def test_trace_capture_plan_fails_if_training_repo_volume_is_short(tmp_path):
    manifest_dir = tmp_path / "manifest"
    tasks_jsonl = manifest_dir / "generated_repo_tasks" / "taskspecs.jsonl"
    _write_jsonl(tasks_jsonl, [_repo_task("generated_repo_tasks__t0")])

    with pytest.raises(ValueError, match="needs 2 eligible"):
        build_trace_capture_plan(
            manifest_dir=manifest_dir,
            tasks_jsonl=tasks_jsonl,
            out_json=manifest_dir / "trace_capture" / "plan.json",
            jobs_out=manifest_dir / "trace_capture" / "jobs.jsonl",
            task_limit=2,
            seed=0,
        )
