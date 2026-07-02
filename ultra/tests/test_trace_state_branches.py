import json

from ultra.trace_state_branches import build_trace_state_branch_report


def test_trace_state_branch_audit_classifies_trace_and_rollout(tmp_path):
    trace_jsonl = tmp_path / "traces.jsonl"
    trace_jsonl.write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "origin_harness": "opencode",
                "worker_model": "kimi",
                "task_id": "repo__bug",
                "repo": {"url": None, "base_commit": None, "initial_tree_hash": None},
                "prompt": {"user_task": "Fix it"},
                "events": [{"type": "test_result", "agent_turn": 1}],
                "artifacts": {},
                "grade": {"score": 0.0, "success": False},
            }
        )
        + "\n"
    )
    rollout_json = tmp_path / "rollout.json"
    rollout_json.write_text(
        json.dumps(
            {
                "rollout_id": "r1",
                "task_id": "training_repo_canary__x",
                "source_name": "training_repo_canary",
                "harness": "opencode",
                "valid_for_training": True,
                "reward": 1.0,
                "grade": {"success": True},
                "execution": {"steps": [{"worker_id": 2, "text": "diff --git a/x b/x\n"}]},
            }
        )
    )

    out_jsonl = tmp_path / "candidates.jsonl"
    report = build_trace_state_branch_report(
        trace_jsonls=[trace_jsonl],
        rollout_jsons=[rollout_json],
        out_jsonl=out_jsonl,
        report_out=tmp_path / "report.json",
    )

    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert report["candidate_count"] == 2
    assert report["train_ready_count"] == 0
    assert {row["state_type"] for row in rows} == {"outcome_only_trace", "post_patch_rollout"}
    assert report["missing_for_training"]["repo_state"] == 1
    assert report["missing_for_training"]["train_allowed_source"] == 1


def test_trace_state_branch_audit_accepts_artifact_backed_trace(tmp_path):
    trace_jsonl = tmp_path / "traces.jsonl"
    trace_jsonl.write_text(
        json.dumps(
            {
                "trace_id": "t-ready",
                "origin_harness": "codex",
                "worker_model": "gpt-5.5",
                "task_id": "generated_repo_tasks__x",
                "repo": {"url": "local://generated_repo_tasks/x", "base_commit": "generated-v1"},
                "prompt": {"user_task": "Fix it"},
                "events": [
                    {"type": "message", "agent_turn": 0, "content_ref": "/tmp/prompt.txt"},
                    {"type": "command", "agent_turn": 0, "content_ref": "/tmp/command.json"},
                    {"type": "file_edit", "agent_turn": 0, "content_ref": "/tmp/patch.diff"},
                    {"type": "test_result", "agent_turn": 0, "content_ref": "/tmp/grade.json"},
                ],
                "artifacts": {
                    "final_patch_ref": "/tmp/patch.diff",
                    "workspace_snapshot_ref": "/tmp/workspace",
                    "public_test_log_ref": "/tmp/command.json",
                    "hidden_grade_ref": "/tmp/grade.json",
                },
                "grade": {"score": 1.0, "success": True},
            }
        )
        + "\n"
    )

    report = build_trace_state_branch_report(
        trace_jsonls=[trace_jsonl],
        rollout_jsons=[],
        out_jsonl=tmp_path / "candidates.jsonl",
        report_out=tmp_path / "report.json",
    )

    rows = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert report["candidate_count"] == 1
    assert report["train_ready_count"] == 1
    assert rows[0]["train_ready"] is True
    assert rows[0]["state_type"] == "trace_checkpoint"
