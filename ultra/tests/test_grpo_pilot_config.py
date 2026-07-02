import json
from pathlib import Path

import yaml

from ultra.grpo_pilot_config import build_grpo_pilot_config


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha(path):
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _task(task_id):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "math",
        "source": {"name": "existing_bank", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Solve"}]},
        "environment": {"harness": "direct_qa"},
        "grader": {"type": "exact_match", "expected_answer": "ok"},
        "splitting": {"group_id": "g", "split": "grpo_train", "contamination_group": task_id},
    }


def test_build_grpo_pilot_config_links_frozen_manifest_and_worker_pool(tmp_path):
    tasks_path = tmp_path / "pilot" / "taskspecs.jsonl"
    seed_path = tmp_path / "pilot" / "seed_manifest.jsonl"
    _write_jsonl(tasks_path, [_task("task-a")])
    _write_jsonl(
        seed_path,
        [
            {
                "lane": "math_science_knowledge",
                "source_task_id": "task-a",
                "selection_reasons": ["reward_variance"],
                "recommended_group_size": 8,
            }
        ],
    )
    freeze_report = {
        "freeze_complete": True,
        "manifests": [
            {
                "manifest_name": "grpo_pilot_tasks",
                "path": str(tasks_path),
                "row_count": 1,
                "sha256": _sha(tasks_path),
                "task_id_sha256": "sha256:x",
            },
            {
                "manifest_name": "grpo_pilot_seed_evidence",
                "path": str(seed_path),
                "row_count": 1,
                "sha256": _sha(seed_path),
                "task_id_sha256": "sha256:x",
            },
        ],
    }
    pool_report = {
        "recommendations": {
            "selection_status": "mvp_grpo_pool_selected_for_pilot_not_final_ultra_claim",
            "recommended_mvp_grpo_workers": ["direct_gpt_reasoner", "direct_gemini_synth"],
            "lane_worker_masks": {"math_science_knowledge": ["direct_gpt_reasoner", "direct_gemini_synth"]},
            "challenger_workers": {"direct_flash_fast": ["observed successes"]},
        },
        "workers": {
            "direct_gpt_reasoner": {"identity": {"name": "direct_gpt_reasoner", "model": "gpt-5.5"}},
            "direct_gemini_synth": {"identity": {"name": "direct_gemini_synth", "model": "gemini-3.1-pro-preview"}},
        },
    }
    freeze_path = tmp_path / "freeze.json"
    pool_path = tmp_path / "pool.json"
    _write_json(freeze_path, freeze_report)
    _write_json(pool_path, pool_report)

    config = build_grpo_pilot_config(
        freeze_report_json=freeze_path,
        pool_report_json=pool_path,
        out_json=tmp_path / "pilot_config.json",
        md_out=tmp_path / "pilot_config.md",
    )

    assert config["ready_for_pilot"] is True
    assert config["task_count"] == 1
    assert config["group_size_by_lane"] == {"math_science_knowledge": 8}
    assert config["lane_worker_masks"]["math_science_knowledge"] == ["direct_gpt_reasoner", "direct_gemini_synth"]
    assert config["challenger_workers_not_in_action_space"] == ["direct_flash_fast"]
    assert config["provider_policy"]["gpt_never_openrouter"] is True
    assert (tmp_path / "pilot_config.md").exists()


def test_parent_repair_tight_slice_configs_use_fresh_parent_base():
    root = Path(__file__).resolve().parents[2]
    manifest_dir = root / "director/manifests/fugu_clean_v1/grpo_pilot_train"
    parent_model = str(root / "output/fugu_ultra_policy_repair_sft_parent_qwen3_8b")
    output_root = str(root / "output/fugu_ultra_grpo_pilot_qwen3_8b_after_parent_repair_sft_tight_repo_tool")

    infer_cfg = yaml.safe_load((manifest_dir / "infer_pilot_qwen3_8b_after_parent_repair_sft.yaml").read_text())
    orch_cfg = yaml.safe_load((manifest_dir / "orch_tight_repo_tool_after_parent_repair_sft_commercial.yaml").read_text())
    train_cfg = yaml.safe_load((manifest_dir / "train_tight_repo_tool_after_parent_repair_sft.yaml").read_text())
    pilot_cfg = json.loads((manifest_dir / "pilot_config_tight_repo_tool_after_parent_repair_sft.json").read_text())
    safety_path = str(manifest_dir / "live_safety_parent_repair_tight_repo_tool.json")

    assert infer_cfg["model"] == parent_model
    assert infer_cfg["enable_lora"] is True
    assert orch_cfg["model"]["name"] == parent_model
    assert train_cfg["model"] == parent_model
    assert train_cfg.get("resume_from_checkpoint") is None
    assert train_cfg["max_steps"] == orch_cfg["max_steps"] == 10
    assert train_cfg["output_dir"] == output_root
    assert orch_cfg["output_dir"] == f"{output_root}/run_default"
    assert {env["args"]["live_safety_path"] for env in orch_cfg["env"]} == {safety_path}
    assert all(env["extra_env_kwargs"]["force_step_budget"] == "short" for env in orch_cfg["env"])
    assert pilot_cfg["checks"]["tight_next_slice"]["base_model"] == "output/fugu_ultra_policy_repair_sft_parent_qwen3_8b"
    assert pilot_cfg["checks"]["tight_next_slice"]["status"] == "prepared_parent_repair_sft_fresh"
