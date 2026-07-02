import io
import json
import tarfile
from pathlib import Path

from ultra.sources import SOURCE_ADAPTERS, HarborTaskBundleAdapter
from ultra.sources.harbor import (
    discover_harbor_task_dirs,
    extract_tasktrove_parquet_bundles,
    harbor_task_to_spec,
    materialize_harbor_tasks,
    materialize_tasktrove_parquet,
)


def _tar_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


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
    report = materialize_harbor_tasks(
        tmp_path,
        out,
        tmp_path / "report.json",
        source_name="tasktrove_inferredbugs",
        policy="train_allowed",
        split="grpo_train",
    )

    assert report["materialized"] == 1
    assert report["policy"] == "train_allowed"
    assert report["split"] == "grpo_train"
    assert out.read_text().count("\n") == 1
    row = json.loads(out.read_text())
    assert row["source"]["name"] == "tasktrove_inferredbugs"
    assert row["source"]["policy"] == "train_allowed"
    assert row["splitting"]["split"] == "grpo_train"


def test_harbor_adapter_is_registered_and_accepts_raw_task_dirs(tmp_path):
    task = _bundle(tmp_path)

    assert SOURCE_ADAPTERS["tasktrove_harbor"] is HarborTaskBundleAdapter
    specs = list(HarborTaskBundleAdapter([{"task_dir": str(task)}]).materialize_all())

    assert len(specs) == 1
    assert specs[0].source.name == "tasktrove_harbor"


def test_tasktrove_parquet_materializes_harbor_bundle(tmp_path):
    parquet = _write_parquet(
        tmp_path / "tasks.parquet",
        [
            {
                "path": "laion/example/tasks/demo-task",
                "task_binary": _tar_bytes(
                    {
                        "instruction.md": "Create /app/out.txt with ok.\n",
                        "task.toml": "\n".join(
                            [
                                'version = "1.0"',
                                "[metadata]",
                                'task_id = "demo-from-parquet"',
                                'category = "filesystem"',
                                "[verifier]",
                                "timeout_sec = 120",
                                "[environment]",
                                'docker_image = "example/harbor-task:latest"',
                            ]
                        ),
                        "tests/test.sh": "#!/usr/bin/env bash\nexit 0\n",
                        "environment/Dockerfile": "FROM ubuntu:24.04\n",
                    }
                ),
            }
        ],
    )
    out = tmp_path / "taskspecs.jsonl"
    report = materialize_tasktrove_parquet(
        parquet,
        tmp_path / "extracted",
        out,
        tmp_path / "report.json",
        source_name="tasktrove_diversity_demo",
        policy="train_allowed",
        split="grpo_train",
    )

    assert report["extraction"]["extracted"] == 1
    assert report["materialization"]["materialized"] == 1
    row = json.loads(out.read_text())
    assert row["task_id"] == "tasktrove_diversity_demo__demo-from-parquet"
    assert row["source"]["policy"] == "train_allowed"
    assert row["environment"]["harness"] == "terminal_sandbox"
    assert row["input"]["assets"][0]["harbor_task"]["task_dir"].endswith("demo-task")


def test_tasktrove_parquet_materializes_only_included_paths(tmp_path):
    def row(path, task_id):
        return {
            "path": path,
            "task_binary": _tar_bytes(
                {
                    "instruction.md": f"Solve {task_id}.\n",
                    "task.toml": "\n".join(
                        [
                            'version = "1.0"',
                            "[metadata]",
                            f'task_id = "{task_id}"',
                            'category = "filesystem"',
                            "[verifier]",
                            "timeout_sec = 120",
                            "[environment]",
                            'docker_image = "example/harbor-task:latest"',
                        ]
                    ),
                    "tests/test.sh": "#!/usr/bin/env bash\nexit 0\n",
                    "environment/Dockerfile": "FROM ubuntu:24.04\n",
                }
            ),
        }

    parquet = _write_parquet(
        tmp_path / "tasks.parquet",
        [
            row("keep-me", "keep-task"),
            row("skip-me", "skip-task"),
        ],
    )
    out = tmp_path / "taskspecs.jsonl"

    report = materialize_tasktrove_parquet(
        parquet,
        tmp_path / "extracted",
        out,
        tmp_path / "report.json",
        source_name="tasktrove_exact",
        include_paths={"keep-me"},
    )

    assert report["include_paths_count"] == 1
    assert report["extraction"]["rows_selected"] == 1
    assert report["materialization"]["materialized"] == 1
    row = json.loads(out.read_text())
    assert row["task_id"] == "tasktrove_exact__keep-task"
    assert "skip-task" not in out.read_text()


def test_tasktrove_parquet_extractor_rejects_unsafe_tar_members(tmp_path):
    parquet = _write_parquet(
        tmp_path / "tasks.parquet",
        [
            {
                "path": "laion/example/tasks/bad-task",
                "task_binary": _tar_bytes({"../escape.txt": "bad"}),
            }
        ],
    )

    report = extract_tasktrove_parquet_bundles(parquet, tmp_path / "extracted")

    assert report["extracted"] == 0
    assert report["skipped"] == 1
    assert "unsafe tar member path" in report["skipped_tasks"][0]["reason"]
    assert not (tmp_path / "escape.txt").exists()
