import json

from ultra.data_recipe import (
    build_source_manifests,
    build_source_registry,
    write_data_recipe_artifacts,
)
from ultra.registry import TaskRegistry
from ultra.schemas import TaskSpec


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_source_registry_locks_critical_policies(tmp_path):
    registry = build_source_registry(tmp_path / "manifest")
    lanes = {lane["source_name"]: lane for lane in registry["lanes"]}

    assert lanes["deep_swe_local"]["policy"] == "final_eval_only"
    assert lanes["deep_swe_local"]["allowed_splits"] == ["final_eval"]
    assert lanes["tasktrove_harbor"]["policy"] == "pool_only"
    assert "grpo_train" not in lanes["tasktrove_harbor"]["allowed_splits"]
    assert lanes["tasktrove_inferredbugs"]["policy"] == "train_allowed"
    assert "grpo_train" in lanes["tasktrove_inferredbugs"]["allowed_splits"]
    assert lanes["existing_bank"]["policy"] == "train_allowed"
    assert registry["grpo_train_mvp_mix"]["tool_dialogue"] == 150
    assert registry["grpo_train_mvp_mix"]["unit_and_scientific_code"] == 225
    assert registry["grpo_train_mvp_mix"]["long_context_memory_planning"] == 125
    assert sum(registry["grpo_train_mvp_mix"].values()) == 1000
    assert registry["fixed_workflow_discovery_gate"]["status"] == "required_before_grpo"
    assert "reward is emitted for success, failure, invalid output, and timeout" in registry["tasktrove_validation_gates"]
    assert any("candidate train distribution" in note for note in registry["grpo_train_mvp_notes"])

    manifests = {m.source_name: m for m in build_source_manifests(tmp_path / "manifest")}
    assert manifests["deep_swe_local"].allowed_uses == ["final_eval"]
    assert "grpo_train" in manifests["existing_bank"].allowed_uses


def test_data_recipe_build_promotes_existing_bank_taskspecs(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _write_jsonl(
        manifest_dir / "manifest.jsonl",
        [
            {
                "task_id": "m1",
                "domain": "math",
                "source": "numina_math",
                "prompt": "2+2?",
                "solution": "4",
                "grader": "math_equal",
                "system": "Answer.",
                "split": "train",
            },
            {
                "task_id": "g1",
                "domain": "general",
                "source": "supergpqa",
                "prompt": "Capital of France?",
                "solution": "Paris",
                "grader": "exact_match",
                "system": "",
                "split": "test",
            },
        ],
    )
    out_dir = tmp_path / "out"

    report = write_data_recipe_artifacts(manifest_dir, out_dir)

    assert report["source_count"] >= 20
    assert (out_dir / "source_manifests.jsonl").exists()
    assert (out_dir / "source_registry.json").exists()
    assert report["existing_bank_report"]["materialized"] == 2

    task_rows = [
        json.loads(line)
        for line in (out_dir / "existing_bank_taskspecs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    specs = [TaskSpec.model_validate(row) for row in task_rows]
    assert [s.splitting.split for s in specs] == ["grpo_train", "online_validation"]
    assert specs[0].splitting.contamination_group == "numina_math::m1"

    manifest_rows = [
        json.loads(line)
        for line in (out_dir / "source_manifests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    existing_manifest = next(row for row in manifest_rows if row["source_name"] == "existing_bank")
    registry = TaskRegistry()
    from ultra.schemas import SourceManifest

    registry.register_manifest(SourceManifest.model_validate(existing_manifest))
    registry.add_many(specs)
    assert len(registry) == 2
