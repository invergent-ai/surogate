import json

from ultra.generated_repo_tasks import (
    SOURCE_NAME,
    TASKS,
    materialize_generated_repo_tasks,
    task_spec,
    write_generated_repo_context,
)
from ultra.registry import TaskRegistry
from ultra.schemas import SourceManifest, TaskSpec


def test_generated_repo_context_and_spec_are_trainable(tmp_path):
    task = TASKS[0]
    task_dir = write_generated_repo_context(tmp_path, task, image_prefix="example/generated")
    spec = task_spec(task, task_dir, image_prefix="example/generated")

    assert (task_dir / "Dockerfile").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert spec.source.name == SOURCE_NAME
    assert spec.source.policy == "train_allowed"
    assert spec.splitting.split == "grpo_train"
    assert spec.environment.harness == "opencode"
    assert spec.input.assets[0]["opencode_instance"]["tests_dir"] == str(task_dir / "tests")
    assert "generated_repo" in spec.metadata.tags


def test_materialize_generated_repo_tasks_without_build_ingests(tmp_path):
    out = tmp_path / "tasks.jsonl"
    report = materialize_generated_repo_tasks(
        work_dir=tmp_path / "work",
        out_jsonl=out,
        report_out=tmp_path / "report.json",
        image_prefix="example/generated",
        build=False,
    )

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = [TaskSpec.model_validate(row) for row in rows]
    assert report["task_count"] == len(TASKS)
    assert report["images_built"] is False
    assert len(specs) == len(TASKS)
    assert {s.source.name for s in specs} == {SOURCE_NAME}
    assert {s.splitting.split for s in specs} == {"grpo_train"}
    assert len({s.splitting.contamination_group for s in specs}) == len(specs)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name=SOURCE_NAME,
            source_type="generated_repo",
            version="v1",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(TASKS)
