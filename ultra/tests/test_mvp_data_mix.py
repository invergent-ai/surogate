import json

import pytest

from ultra.mvp_data_mix import build_mvp_grpo_mix


def _task(task_id, source, harness, capability, domain, split="grpo_train", policy="train_allowed"):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": capability,
        "source": {"name": source, "version": "v1", "policy": policy},
        "input": {"messages": [{"role": "user", "content": f"Task {task_id}"}]},
        "environment": {"harness": harness},
        "grader": {"type": "exact_match", "expected_answer": "ok"},
        "splitting": {
            "group_id": source,
            "split": split,
            "contamination_group": f"{source}::{task_id}",
        },
        "metadata": {"domain": domain},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rows(prefix, count, source, harness, capability, domain):
    return [_task(f"{prefix}-{i:04d}", source, harness, capability, domain) for i in range(count)]


def _populate_manifest(manifest_dir, *, pymethods_count=134):
    existing = []
    existing += _rows("code", 300, "existing_bank", "code_exec", "unit_code", "code")
    existing += _rows("math", 100, "existing_bank", "direct_qa", "math", "math")
    existing += _rows("science", 100, "existing_bank", "direct_qa", "science_knowledge", "science")
    existing += _rows("general", 100, "existing_bank", "direct_qa", "factual_qa", "general")
    existing.append(
        _task(
            "deep-swe-decoy",
            "deep_swe_local",
            "opencode",
            "agentic_coding",
            "software_engineering",
            split="final_eval",
            policy="final_eval_only",
        )
    )
    _write_jsonl(manifest_dir / "data_mix" / "existing_bank_taskspecs.jsonl", existing)
    _write_jsonl(
        manifest_dir / "generated_repo_tasks" / "taskspecs.jsonl",
        _rows("repo", 16, "generated_repo_tasks", "opencode", "agentic_coding", "software_engineering"),
    )
    _write_jsonl(
        manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl",
        _rows("inferred", 100, "tasktrove_inferredbugs", "terminal_sandbox", "terminal_agentic", "terminal"),
    )
    _write_jsonl(
        manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl",
        _rows("pymethods", pymethods_count, "tasktrove_pymethods2test", "terminal_sandbox", "terminal_agentic", "terminal"),
    )
    _write_jsonl(
        manifest_dir / "tool_dialog_tasks" / "taskspecs.jsonl",
        _rows("tool", 150, "tau_custom", "tool_dialog", "tool_dialogue", "retail"),
    )
    _write_jsonl(
        manifest_dir / "long_context_tasks" / "taskspecs.jsonl",
        _rows("long", 125, "longctx_generated", "long_context", "long_context", "long_context"),
    )


def test_build_mvp_grpo_mix_writes_exact_source_quotas(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _populate_manifest(manifest_dir)

    out_jsonl = manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl"
    report_path = manifest_dir / "data_mix" / "mvp_grpo_train_report.json"
    report = build_mvp_grpo_mix(
        manifest_dir=manifest_dir,
        out_jsonl=out_jsonl,
        report_out=report_path,
        seed=7,
    )

    assert out_jsonl.exists()
    assert report_path.exists()
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 1000
    assert report["selected_total"] == 1000
    assert report["status"] == "candidate_pending_fixed_workflow_discovery"
    assert report["purpose"] == "fixed_workflow_discovery_and_grpo_pilot_sampling_candidate"
    assert report["lane_counts"] == {
        "long_context_memory_planning": 125,
        "math_science_knowledge": 250,
        "repo_repair_open_repo_terminal": 250,
        "tool_dialogue": 150,
        "unit_and_scientific_code": 225,
    }
    assert report["counts"]["sources"] == {
        "existing_bank": 475,
        "generated_repo_tasks": 16,
        "longctx_generated": 125,
        "tasktrove_inferredbugs": 100,
        "tasktrove_pymethods2test": 134,
        "tau_custom": 150,
    }
    assert report["counts"]["domains"] == {
        "code": 225,
        "general": 83,
        "long_context": 125,
        "math": 84,
        "retail": 150,
        "science": 83,
        "software_engineering": 16,
        "terminal": 234,
    }
    assert report["counts"]["splits"] == {"grpo_train": 1000}
    assert "deep_swe_local" not in report["counts"]["sources"]
    assert all(row["source"]["policy"] == "train_allowed" for row in rows)


def test_build_mvp_grpo_mix_fails_when_a_fixed_source_quota_is_short(tmp_path):
    manifest_dir = tmp_path / "manifest"
    _populate_manifest(manifest_dir, pymethods_count=133)

    with pytest.raises(ValueError, match="tasktrove_pymethods2test"):
        build_mvp_grpo_mix(
            manifest_dir=manifest_dir,
            out_jsonl=manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl",
            seed=0,
        )
