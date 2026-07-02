import json

from ultra.scaffold_materialize import deep_swe_task_to_spec, materialize_repo_tasks


def _deep_swe_task_dir(tmp_path):
    task_dir = tmp_path / "director" / "vendor" / "deep_swe" / "tasks" / "local-task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Fix the local repo bug.\n")
    (task_dir / "tests" / "config.json").write_text('{"base_commit":"abc123"}\n')
    (task_dir / "tests" / "test.patch").write_text("diff --git a/test b/test\n")
    (task_dir / "task.toml").write_text(
        """
schema_version = "1.1"
[metadata]
task_id = "local-task"
display_title = "Local task"
category = "feature_request"
language = "python"
repository_url = "https://github.com/example/repo"
base_commit_hash = "abc123"
[verifier]
timeout_sec = 1800.0
[environment]
docker_image = "public.ecr.aws/example/local-task:v1"
cpus = 2
memory_mb = 4096
storage_mb = 8192
"""
    )
    return task_dir


def test_deep_swe_task_to_spec_materializes_opencode_payload(tmp_path):
    task_dir = _deep_swe_task_dir(tmp_path)
    spec = deep_swe_task_to_spec(
        {
            "source": "deep_swe_local",
            "source_task_id": "local-task",
            "task_dir": str(task_dir),
            "selection_tags": ["deep_swe_local", "python"],
        }
    )
    assert spec is not None
    assert spec.task_id == "deep_swe_local__local-task"
    assert spec.environment.harness == "opencode"
    assert spec.environment.image == "public.ecr.aws/example/local-task:v1"
    assert spec.grader.type == "deep_swe_hidden_tests"
    assert spec.input.repo.url == "https://github.com/example/repo"
    assert spec.input.assets[0]["opencode_instance"]["image_name"] == "public.ecr.aws/example/local-task:v1"
    assert spec.input.assets[0]["opencode_instance"]["testbed"] == "/app"
    assert spec.input.assets[0]["opencode_instance"]["activate"] == ""
    assert spec.source.policy == "final_eval_only"
    assert spec.splitting.split == "final_eval"


def test_materialize_repo_tasks_writes_specs_and_unresolved_report(tmp_path):
    task_dir = _deep_swe_task_dir(tmp_path)
    manifest = {
        "tasks": [
            {
                "domain": "coding_repo",
                "source": "deep_swe_local",
                "source_task_id": "local-task",
                "task_dir": str(task_dir),
                "selection_tags": ["deep_swe_local", "python"],
            },
            {
                "domain": "coding_repo",
                "source": "agentic_coding_frontier_direct3",
                "source_task_id": "pydicom__missing_payload",
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    out_jsonl = tmp_path / "taskspecs.jsonl"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest))

    report = materialize_repo_tasks(manifest_path, out_jsonl, report_path)

    assert report["materialized"] == 1
    assert report["unresolved"] == 1
    assert report["unresolved_tasks"][0]["reason"] == "saved_live_coding_payload_not_available"
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_id"] == "deep_swe_local__local-task"
    assert report_path.exists()
