import json

from ultra.harness_parity import build_harness_parity_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _trace(harness, model):
    return {
        "trace_id": f"{harness}-trace",
        "origin_harness": harness,
        "worker_model": model,
        "events": [
            {"type": "message"},
            {"type": "command"},
            {"type": "file_edit"},
            {"type": "test_result"},
        ],
        "artifacts": {
            "final_patch_ref": "/tmp/patch.diff",
            "workspace_snapshot_ref": "/tmp/workspace",
            "public_test_log_ref": "/tmp/test.log",
            "hidden_grade_ref": "/tmp/grade.json",
        },
        "grade": {"success": True, "score": 1.0},
        "usage": {"cost_usd": 0.0},
    }


def _rollout(step_harness):
    return {
        "grade": {"success": True, "score": 1.0},
        "reward": 1.0,
        "execution": {
            "steps": [
                {
                    "harness": step_harness,
                    "termination": "completed",
                    "text": "diff --git a/a.py b/a.py\n",
                }
            ]
        },
    }


def _harbor_result(reward):
    return {
        "stats": {
            "cost_usd": 0.1,
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "evals": {
                "terminus-2__gpt-5.5__adhoc": {
                    "reward_stats": {"reward": {str(reward): ["task-a"]}},
                }
            },
        }
    }


def test_harness_parity_report_summarizes_saved_artifacts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    repo_root = tmp_path / "repo"
    trace_dir = manifest_dir / "trace_capture" / "agent_traces"
    for harness, model in [
        ("opencode", "moonshotai/kimi-k2.7-code"),
        ("codex", "gpt-5.5"),
        ("claude_code", "claude-opus-4.8"),
    ]:
        _write_json(trace_dir / f"{harness}.json", _trace(harness, model))

    _write_json(manifest_dir / "canaries" / "opencode_kimi_training_slugkit.json", _rollout("opencode"))
    _write_json(
        manifest_dir / "canaries" / "opencode_kimi_training_slugkit_patch_grade.json",
        {"success": True, "reward": 1.0, "raw_diff_len": 100, "sanitized_diff_len": 80},
    )
    _write_json(manifest_dir / "canaries" / "codex_gpt55_yunwu_training_slugkit.json", _rollout("codex"))
    _write_json(
        manifest_dir / "canaries" / "claude_code_opus_yunwu_training_slugkit.json",
        _rollout("claude_code"),
    )
    _write_json(
        manifest_dir / "tasktrove_harbor" / "harbor_jobs" / "fugu_tasktrove_nop_canary" / "result.json",
        _harbor_result(0.0),
    )
    _write_json(
        manifest_dir
        / "tasktrove_harbor"
        / "harbor_jobs"
        / "fugu_tasktrove_model_canary_yunwu_gpt55_0011"
        / "result.json",
        _harbor_result(1.0),
    )
    (manifest_dir / "tasktrove_harbor" / "harbor_jobs" / "fugu_tasktrove_model_canary_yunwu_gpt55_0011").mkdir(
        parents=True, exist_ok=True
    )
    (
        manifest_dir / "tasktrove_harbor" / "harbor_jobs" / "fugu_tasktrove_model_canary_yunwu_gpt55_0011" / "job.log"
    ).write_text("ok")

    report = build_harness_parity_report(
        manifest_dir=manifest_dir,
        repo_root=repo_root,
        report_out=tmp_path / "report.json",
        md_out=tmp_path / "report.md",
        created_at_utc="2026-06-27T00:00:00Z",
    )

    assert report["parity_complete"] is True
    assert {item["harness"] for item in report["harnesses"]} == {
        "opencode",
        "codex_yunwu",
        "claude_code_yunwu_bridge",
        "terminal_sandbox",
        "direct_qa",
        "tool_dialogue",
        "long_context",
    }
    assert all(item["status"] == "pass" for item in report["harnesses"])
    assert report["trace_capture_summary"]["by_harness"]["codex"]["required_artifact_count"] == 1
    assert (tmp_path / "report.json").exists()
    assert "Harness Parity Report" in (tmp_path / "report.md").read_text()
