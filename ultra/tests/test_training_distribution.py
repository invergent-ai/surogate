import json

from ultra.training_distribution import build_training_distribution_plan, write_training_distribution_plan


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_training_distribution_keeps_deep_swe_eval_only(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _write_jsonl(
        manifest_dir / "scaffold_repo_taskspecs.jsonl",
        [
            {
                "task_id": "deep_swe_local__a",
                "capability": "agentic_coding",
                "source": {"name": "deep_swe_local", "policy": "final_eval_only"},
                "environment": {"harness": "opencode"},
                "splitting": {"split": "final_eval"},
                "metadata": {"domain": "software_engineering"},
            }
        ],
    )

    plan = build_training_distribution_plan(manifest_dir)

    assert sum(tier["share"] for tier in plan["rollout_mix"]) == 1.0
    assert plan["mvp_task_mix_status"] == "candidate_pending_fixed_workflow_discovery"
    assert plan["fixed_workflow_discovery_gate"]["status"] == "required_before_grpo"
    assert plan["manifest_freeze_requirements"]["status"] == "required_before_fixed_workflow_discovery"
    assert "claude_code_local_bridge_yunwu_opus_4_8" in plan["harness_parity_canaries"]["harnesses"]
    assert plan["failure_taxonomy"]["provider_failure_retry_or_exclude"]["use"] == "retry_or_exclude"
    assert "source_validation_report.json" in plan["source_validation_report_spec"]["required_outputs"]
    assert plan["fixed_workflow_discovery_gate"]["lane_mix"]["repo_repair_open_repo_terminal"] == 50
    assert "codex:yunwu-gpt-5.5" in plan["worker_masks"]["repo_coding"]["candidates"]
    assert "best individual model+scaffold selected on dev" in plan["evaluation_baselines"]
    assert any("within-task reward variance" in gate for gate in plan["go_no_go_gates"])
    assert "deep_swe_local" in plan["canary_distribution"]["forbidden_sources"]
    assert all(
        "Deep SWE" not in source and "deep_swe" not in source.lower()
        for tier in plan["rollout_mix"]
        for source in tier["sources"]
    )
    assert plan["artifacts"]["deep_swe_eval_taskspecs"]["policies"] == {"final_eval_only": 1}
    assert plan["artifacts"]["deep_swe_eval_taskspecs"]["splits"] == {"final_eval": 1}


def test_training_distribution_counts_current_artifacts_and_writes(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _write_jsonl(
        manifest_dir / "training_repo_canaries" / "taskspecs.jsonl",
        [
            {
                "task_id": "training_repo_canary__slugkit",
                "capability": "agentic_coding",
                "source": {"name": "training_repo_canary", "policy": "train_allowed"},
                "environment": {"harness": "opencode"},
                "splitting": {"split": "diagnostic"},
                "metadata": {"domain": "software_engineering"},
            }
        ],
    )
    out_json = tmp_path / "out" / "plan.json"
    out_md = tmp_path / "out" / "plan.md"

    plan = write_training_distribution_plan(manifest_dir, out_json, out_md)

    assert out_json.exists()
    assert out_md.exists()
    assert json.loads(out_json.read_text())["version"] == "fugu_ultra_training_distribution_v1"
    assert "Fugu-Ultra Training Task Distribution" in out_md.read_text()
    assert "Fixed-Workflow Discovery Gate" in out_md.read_text()
    assert "Manifest Freeze Requirements" in out_md.read_text()
    assert "Failure Taxonomy" in out_md.read_text()
    assert plan["artifacts"]["training_repo_canary_taskspecs"]["count"] == 1
