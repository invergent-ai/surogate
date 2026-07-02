import json

from ultra.source_validation import build_source_validation_report


def _task(task_id, source, harness, capability, domain, *, assets=None, context=None, tools=None, expected="ok"):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": capability,
        "source": {"name": source, "version": "v1", "policy": "train_allowed"},
        "input": {
            "messages": [{"role": "user", "content": f"Task {task_id}"}],
            "assets": assets or [],
            "context_documents": context or [],
            "tools": tools or [],
        },
        "environment": {"harness": harness, "wall_time_seconds": 120},
        "grader": {"type": "exact_match", "expected_answer": expected, "deterministic": True},
        "splitting": {
            "group_id": task_id,
            "split": "grpo_train",
            "contamination_group": f"{source}/{task_id}",
        },
        "metadata": {"domain": domain, "estimated_worker_calls": 1},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_source_validation_writes_reports_and_parquet(tmp_path):
    manifest_dir = tmp_path / "manifest"
    repo_task = manifest_dir / "generated_repo_tasks" / "repo-a"
    tests_dir = repo_task / "tests"
    tests_dir.mkdir(parents=True)
    (repo_task / "instruction.md").write_text("Fix it")
    (tests_dir / "test.sh").write_text("exit 0")

    rows = [
        _task("math-1", "existing_bank", "direct_qa", "math", "math"),
        _task("code-1", "existing_bank", "code_exec", "unit_code", "code"),
        _task(
            "repo-1",
            "generated_repo_tasks",
            "opencode",
            "agentic_coding",
            "software_engineering",
            assets=[
                {
                    "opencode_instance": {
                        "image_name": "repo-a:latest",
                        "task_dir": str(repo_task),
                        "tests_dir": str(tests_dir),
                    }
                }
            ],
            expected=None,
        ),
        _task(
            "tool-1",
            "tau_custom",
            "tool_dialog",
            "tool_dialogue",
            "retail",
            tools=[{"type": "function", "function": {"name": "finish", "parameters": {}}}],
            expected={"success": [{"path": ["x"], "equals": 1}]},
        ),
        _task(
            "long-1",
            "longctx_generated",
            "long_context",
            "long_context",
            "long_context",
            context=[{"title": "doc", "text": "answer ok"}],
        ),
    ]
    tasks_jsonl = manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl"
    _write_jsonl(tasks_jsonl, rows)
    (manifest_dir / "generated_repo_tasks").mkdir(exist_ok=True)
    (manifest_dir / "generated_repo_tasks" / "report.json").write_text(
        json.dumps({"base_validation_ready": True})
    )
    (manifest_dir / "frozen_manifests").mkdir()
    _write_jsonl(manifest_dir / "frozen_manifests" / "final_eval.jsonl", [])
    _write_jsonl(manifest_dir / "frozen_manifests" / "deep_swe_target_eval.jsonl", [])

    report = build_source_validation_report(
        manifest_dir=manifest_dir,
        tasks_jsonl=tasks_jsonl,
        report_out=manifest_dir / "source_validation_report.json",
        md_out=manifest_dir / "source_validation_report.md",
        difficulty_out=manifest_dir / "difficulty_calibration.parquet",
        quality_flags_out=manifest_dir / "task_quality_flags.parquet",
        created_at_utc="2026-06-27T00:00:00Z",
    )

    assert report["status"] == "pass"
    assert report["task_count"] == 5
    assert report["valid_for_grpo"] == 5
    assert report["counts"]["difficulty"]["medium"] >= 3
    assert (manifest_dir / "source_validation_report.json").exists()
    assert "Source Validation" in (manifest_dir / "source_validation_report.md").read_text()
    assert report["artifacts"]["task_quality_flags"]["rows"] == 5
    assert report["artifacts"]["difficulty_calibration"]["rows"] >= 3
    if report["artifacts"]["task_quality_flags"]["written"]:
        assert (manifest_dir / "task_quality_flags.parquet").exists()
    else:
        assert (manifest_dir / "task_quality_flags.parquet.jsonl").exists()
