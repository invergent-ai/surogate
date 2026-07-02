import json
from pathlib import Path

from ultra.trace_branch_tasks import materialize_trace_branch_tasks
from ultra.schemas import EnvironmentSpec, GraderSpec, SourceRef, SplittingSpec, TaskInput, TaskSpec


def _base_task() -> TaskSpec:
    return TaskSpec(
        task_id="generated_repo_tasks__x",
        capability="agentic_coding",
        source=SourceRef(name="generated_repo_tasks", version="v1", policy="train_allowed"),
        input=TaskInput(
            messages=[{"role": "user", "content": "Fix x"}],
            assets=[
                {
                    "opencode_instance": {
                        "image_name": "example/x:latest",
                        "instance_id": "",
                        "problem_statement": "Fix x",
                        "testbed": "/app",
                    }
                }
            ],
        ),
        environment=EnvironmentSpec(harness="opencode", wall_time_seconds=900),
        grader=GraderSpec(type="deep_swe_hidden_tests"),
        splitting=SplittingSpec(
            group_id="generated_repo_tasks/x",
            split="grpo_train",
            contamination_group="generated_repo_tasks/x",
        ),
    )


def test_materialize_trace_branch_tasks_writes_initial_patch_taskspec(tmp_path):
    base_tasks = tmp_path / "base.jsonl"
    base_tasks.write_text(json.dumps(_base_task().model_dump(mode="json")) + "\n")

    patch = tmp_path / "patch.diff"
    patch.write_text("diff --git a/x.py b/x.py\n")
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "origin_harness": "codex",
                "worker_model": "gpt-5.5",
                "task_id": "generated_repo_tasks__x",
                "repo": {"url": "local://x", "base_commit": "base"},
                "prompt": {"user_task": "Fix x"},
                "events": [{"type": "file_edit", "agent_turn": 0, "content_ref": str(patch)}],
                "artifacts": {
                    "final_patch_ref": str(patch),
                    "workspace_snapshot_ref": str(tmp_path / "workspace"),
                    "hidden_grade_ref": str(tmp_path / "grade.json"),
                    "public_test_log_ref": str(tmp_path / "command.json"),
                },
                "grade": {"score": 0.0, "success": False},
            }
        )
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "branch_123",
                "source_kind": "agent_trace",
                "source_path": str(trace),
                "task_id": "generated_repo_tasks__x",
                "origin_harness": "codex",
                "worker_model": "gpt-5.5",
                "state_type": "trace_checkpoint",
                "train_ready": True,
            }
        )
        + "\n"
    )

    report = materialize_trace_branch_tasks(
        branch_candidates_jsonl=candidates,
        base_tasks_jsonl=base_tasks,
        out_jsonl=tmp_path / "branch_tasks.jsonl",
        report_out=tmp_path / "report.json",
    )

    row = json.loads((tmp_path / "branch_tasks.jsonl").read_text())
    task = TaskSpec.model_validate(row)
    instance = task.input.assets[0]["opencode_instance"]
    branch = task.input.assets[1]["trace_branch"]
    assert report["materialized"] == 1
    assert task.source.name == "trace_state_branches"
    assert task.splitting.split == "grpo_train"
    assert instance["initial_patch_ref"] == str(patch)
    assert "prior patch already applied" in instance["problem_statement"]
    assert branch["source_trace_ref"] == str(trace)
    assert "origin_harness:codex" in task.metadata.tags


def test_materialize_trace_branch_tasks_accepts_rollout_records(tmp_path):
    base_tasks = tmp_path / "base.jsonl"
    base_tasks.write_text(json.dumps(_base_task().model_dump(mode="json")) + "\n")

    patch = tmp_path / "patch.diff"
    patch.write_text("diff --git a/x.py b/x.py\n")
    workspace = tmp_path / "workspace"
    command = tmp_path / "command.json"
    grade = tmp_path / "grade.txt"
    rollout = tmp_path / "rollout.json"
    rollout.write_text(
        json.dumps(
            {
                "rollout_id": "rollout-1",
                "task_id": "generated_repo_tasks__x",
                "source_name": "generated_repo_tasks",
                "capability": "agentic_coding",
                "harness": "opencode",
                "workflow": {
                    "steps": [
                        {
                            "worker_id": 2,
                            "subtask": "Fix x",
                            "access": [],
                            "budget": "short",
                        }
                    ]
                },
                "execution": {
                    "steps": [
                        {
                            "worker_id": 2,
                            "harness": "claude_code",
                            "budget": "short",
                            "patch_ref": str(patch),
                            "session_ref": str(workspace),
                            "tool_events_ref": str(command),
                            "text": patch.read_text(),
                        }
                    ]
                },
                "grade": {
                    "score": 1.0,
                    "success": True,
                    "details": {
                        "hidden_grade_ref": str(grade),
                        "public_test_log_ref": str(command),
                    },
                },
                "reward": 1.0,
                "valid_for_training": True,
            }
        )
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "branch_rollout",
                "source_kind": "rollout_record",
                "source_path": str(rollout),
                "task_id": "generated_repo_tasks__x",
                "origin_harness": "opencode",
                "worker_ids": [2],
                "state_type": "post_patch_rollout",
                "train_ready": True,
            }
        )
        + "\n"
    )

    report = materialize_trace_branch_tasks(
        branch_candidates_jsonl=candidates,
        base_tasks_jsonl=base_tasks,
        out_jsonl=tmp_path / "branch_tasks.jsonl",
        report_out=tmp_path / "report.json",
    )

    row = json.loads((tmp_path / "branch_tasks.jsonl").read_text())
    task = TaskSpec.model_validate(row)
    instance = task.input.assets[0]["opencode_instance"]
    branch = task.input.assets[1]["trace_branch"]
    assert report["materialized"] == 1
    assert instance["initial_patch_ref"] == str(patch)
    assert instance["workspace_snapshot_ref"] == str(workspace)
    assert branch["source_kind"] == "rollout_record"
    assert branch["public_test_log_ref"] == str(command)
    assert branch["previous_success"] is True
    assert "origin_harness:claude_code" in task.metadata.tags
