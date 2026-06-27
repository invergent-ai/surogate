from pathlib import Path

from ultra.sources import SOURCE_ADAPTERS, HarborTaskBundleAdapter
from ultra.sources.harbor import discover_harbor_task_dirs, harbor_task_to_spec, materialize_harbor_tasks


def _bundle(root: Path, name: str = "task-a") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Create /app/out.txt with ok.\n")
    (task / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "[metadata]",
                'task_id = "demo-task"',
                'difficulty = "medium"',
                'category = "filesystem"',
                'tags = ["file", "shell"]',
                "[verifier]",
                "timeout_sec = 120",
                "[agent]",
                "timeout_sec = 180",
                "[environment]",
                'docker_image = "example/harbor-task:latest"',
                "cpus = 2",
                "memory_mb = 1024",
            ]
        )
    )
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    return task


def test_harbor_task_bundle_materializes_to_terminal_sandbox(tmp_path):
    task = _bundle(tmp_path)

    spec = harbor_task_to_spec(task)

    assert spec is not None
    assert spec.task_id == "tasktrove_harbor__demo-task"
    assert spec.environment.harness == "terminal_sandbox"
    assert spec.environment.image == "example/harbor-task:latest"
    assert spec.grader.type == "harbor_verifier"
    assert spec.input.assets[0]["harbor_task"]["task_dir"] == str(task.resolve())
    assert "verifier_backed" in spec.metadata.tags


def test_harbor_discovery_and_materialization_report(tmp_path):
    _bundle(tmp_path, "task-a")
    no_verifier = tmp_path / "task-b"
    no_verifier.mkdir()
    (no_verifier / "instruction.md").write_text("No verifier\n")
    (no_verifier / "task.toml").write_text('version = "1.0"\n')

    found = discover_harbor_task_dirs(tmp_path)
    assert [p.name for p in found] == ["task-a"]

    out = tmp_path / "specs.jsonl"
    report = materialize_harbor_tasks(tmp_path, out, tmp_path / "report.json")

    assert report["materialized"] == 1
    assert out.read_text().count("\n") == 1


def test_harbor_adapter_is_registered_and_accepts_raw_task_dirs(tmp_path):
    task = _bundle(tmp_path)

    assert SOURCE_ADAPTERS["tasktrove_harbor"] is HarborTaskBundleAdapter
    specs = list(HarborTaskBundleAdapter([{"task_dir": str(task)}]).materialize_all())

    assert len(specs) == 1
    assert specs[0].source.name == "tasktrove_harbor"
