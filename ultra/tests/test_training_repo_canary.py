import json

from ultra.training_repo_canary import (
    DEFAULT_IMAGE_TAG,
    materialize_training_repo_canaries,
    slugkit_spec,
    write_slugkit_context,
)


def test_slugkit_context_and_spec_are_training_distribution(tmp_path):
    task_dir = write_slugkit_context(tmp_path, image_tag="example/slugkit:test")
    spec = slugkit_spec(task_dir, image_tag="example/slugkit:test")

    assert (task_dir / "Dockerfile").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert spec.source.name == "training_repo_canary"
    assert spec.source.policy == "train_allowed"
    assert spec.splitting.split == "diagnostic"
    assert spec.environment.harness == "opencode"
    assert spec.environment.image == "example/slugkit:test"
    assert "training_distribution" in spec.metadata.tags


def test_materialize_training_repo_canaries_without_build(tmp_path):
    out = tmp_path / "tasks.jsonl"
    report = materialize_training_repo_canaries(
        work_dir=tmp_path / "work",
        out_jsonl=out,
        report_out=tmp_path / "report.json",
        build=False,
    )

    row = json.loads(out.read_text())
    assert report["image_built"] is False
    assert report["policy"] == "train_allowed"
    assert row["task_id"] == "training_repo_canary__slugkit-normalize-title"
    assert row["input"]["assets"][0]["opencode_instance"]["image_name"] == DEFAULT_IMAGE_TAG
