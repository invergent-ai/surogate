import json
import sys

from surogate.regression import baseline_runner as br


def test_runner_uses_config_model_without_local_override(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN3_MODEL_PATH", raising=False)

    result = br.run_case(br.REGRESSION_MATRIX[0], run=False, steps=1)

    assert result.status == "skipped"
    assert result.reason == "not executed; pass --run to launch GPU workload"


def test_compare_detects_status_drift():
    case = br.REGRESSION_MATRIX[0]
    base = {case.case_id: {"case": br.asdict(case), "status": "passed"}}
    cur = {case.case_id: {"case": br.asdict(case), "status": "failed"}}

    failures = br.compare_results(cur, base)

    assert failures == [f"{case.case_id}: status failed != passed"]


def test_coverage_report_counts_supported_quant_rows(tmp_path):
    fp8_case = br.RegressionCase("m", "fp8", "single_gpu")
    fp4_unsupported = br.RegressionCase("m", "fp4", "single_gpu", supported=False)
    results = {
        fp8_case.case_id: {"case": br.asdict(fp8_case), "status": "passed"},
        fp4_unsupported.case_id: {"case": br.asdict(fp4_unsupported), "status": "skipped"},
    }

    report = br.coverage_report(results)

    assert report["eligible"] == 1
    assert report["passed"] == 1
    assert report["coverage"] == 1.0
    assert report["rows"][0]["required_capabilities"] == ["DenseMatmul", "FP8Eligible"]
    assert report["rows"][0]["required_matmul_capabilities"] == ["FP8ForwardEligible", "FP8BackwardEligible"]
    assert report["rows"][0]["descriptor_requirement_status"] == "unknown"
    assert report["rows"][0]["missing_descriptor_counts"] == [
        "matmul_fp8_forward_eligible_ops",
        "matmul_fp8_backward_eligible_ops",
    ]
    assert report["rows"][0]["hook_readiness_status"] == "not_applicable"
    assert report["rows"][0]["missing_hook_counts"] == []
    assert report["rows"][1]["descriptor_requirement_status"] == "not_applicable"
    assert report["rows"][1]["missing_descriptor_counts"] == []
    assert json.dumps(report)


def test_coverage_report_marks_moe_grouped_capabilities():
    case = br.RegressionCase("m", "fp8", "2gpu_dp_ep", storage="cpu_stream", op_kind="moe_grouped")
    results = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "skipped",
            "reason": "missing weights",
            "metrics": {
                "descriptor_summary": {
                    "matmul_fp8_forward_eligible_ops": 8,
                    "matmul_fp8_backward_eligible_ops": 8,
                    "moe_fp8_grouped_eligible_ops": 2,
                    "fusion_candidate_starts": 3,
                    "forward_matmul_fp8_forward_eligible_ops": 4,
                    "backward_matmul_fp8_backward_eligible_ops": 4,
                    "lora_slices": 2,
                    "lora_schema_slot_slices": 2,
                    "lora_schema_target_slices": 2,
                    "grouped_lora_schema_slot_slices": 2,
                    "grouped_lora_schema_target_slices": 2,
                    "forward_hook_points": 2,
                    "forward_hook_schema_slot_points": 2,
                    "forward_hook_schema_target_points": 2,
                },
                "block_schema_summary": {
                    "block_schema_records": 2,
                    "block_schema_expected_layers": 2,
                    "block_schema_missing_layers": 0,
                    "block_schema_moe_layers": 2,
                    "block_schema_ep_layers": 2,
                    "block_schema_auto_resident_slots": 4,
                    "hook_after_produce_targets": 2,
                    "hook_before_consume_targets": 4,
                    "hook_after_consume_targets": 4,
                    "hook_after_communication_targets": 2,
                    "hook_after_all_to_all_targets": 2,
                    "hook_after_all_reduce_targets": 1,
                    "hook_after_reduce_scatter_targets": 2,
                },
                "buffer_plan_summary": {
                    "schema_record_count": 2,
                    "schema_expert_parallel_param_shape_savings_bytes": 1024,
                    "hook_registry_registrations": 12,
                    "schema_hook_dispatch_enabled": 1,
                    "hook_registry_distribution_aware_registrations": 4,
                    "hook_registry_after_produce_registrations": 2,
                    "hook_registry_before_consume_registrations": 2,
                    "hook_registry_after_consume_registrations": 2,
                    "hook_registry_after_communication_registrations": 2,
                    "hook_registry_after_all_reduce_registrations": 1,
                    "hook_registry_after_all_to_all_registrations": 2,
                    "hook_registry_after_reduce_scatter_registrations": 1,
                },
            },
        }
    }

    report = br.coverage_report(results)

    assert report["rows"][0]["required_capabilities"] == ["GroupedMatmul", "MoERouted", "FP8Eligible"]
    assert report["rows"][0]["required_moe_capabilities"] == ["GroupedGemmEligible", "FP8GroupedEligible"]
    assert report["rows"][0]["required_matmul_capabilities"] == []
    assert report["rows"][0]["matmul_capability_counts"]["matmul_fp8_forward_eligible_ops"] == 8
    assert report["rows"][0]["matmul_capability_counts"]["matmul_fp8_backward_eligible_ops"] == 8
    assert report["rows"][0]["matmul_capability_counts"]["forward_matmul_fp8_forward_eligible_ops"] == 4
    assert report["rows"][0]["matmul_capability_counts"]["backward_matmul_fp8_backward_eligible_ops"] == 4
    assert report["rows"][0]["descriptor_requirement_status"] == "present"
    assert report["rows"][0]["missing_descriptor_counts"] == []
    assert report["rows"][0]["fusion_candidate_starts"] == 3
    assert report["rows"][0]["block_schema_status"] == "present"
    assert report["rows"][0]["storage_declaration_status"] == "present"
    assert report["rows"][0]["ep_topology_status"] == "present"
    assert report["rows"][0]["hook_readiness_status"] == "present"
    assert report["rows"][0]["missing_hook_counts"] == []
    assert report["rows"][0]["hook_target_counts"]["hook_after_produce_targets"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_after_consume_targets"] == 4
    assert report["rows"][0]["hook_target_counts"]["hook_after_communication_targets"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_after_all_to_all_targets"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_after_all_reduce_targets"] == 1
    assert report["rows"][0]["hook_target_counts"]["lora_schema_slot_slices"] == 2
    assert report["rows"][0]["hook_target_counts"]["lora_schema_target_slices"] == 2
    assert report["rows"][0]["hook_target_counts"]["forward_hook_schema_slot_points"] == 2
    assert report["rows"][0]["hook_target_counts"]["forward_hook_schema_target_points"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_registry_registrations"] == 12
    assert report["rows"][0]["hook_target_counts"]["schema_hook_dispatch_enabled"] == 1
    assert report["rows"][0]["hook_target_counts"]["hook_registry_after_produce_registrations"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_registry_after_consume_registrations"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_registry_after_communication_registrations"] == 2
    assert report["rows"][0]["hook_target_counts"]["hook_registry_after_all_to_all_registrations"] == 2
    assert report["rows"][0]["block_schema_summary"]["block_schema_moe_layers"] == 2
    assert report["rows"][0]["buffer_plan_summary"]["schema_expert_parallel_param_shape_savings_bytes"] == 1024


def test_coverage_report_marks_missing_storage_and_ep_schema_statuses():
    case = br.RegressionCase("m", "fp8", "2gpu_dp_ep", storage="cpu_stream", op_kind="moe_grouped")
    results = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "failed",
            "metrics": {
                "block_schema_summary": {
                    "block_schema_records": 1,
                    "block_schema_expected_layers": 1,
                    "block_schema_missing_layers": 0,
                }
            },
        }
    }

    report = br.coverage_report(results)

    assert report["rows"][0]["block_schema_status"] == "present"
    assert report["rows"][0]["storage_declaration_status"] == "missing"
    assert report["rows"][0]["ep_topology_status"] == "missing"
    assert report["rows"][0]["hook_readiness_status"] == "missing"
    assert report["rows"][0]["missing_hook_counts"] == [
        "hook_before_consume_targets",
        "hook_registry_before_consume_registrations",
        "hook_after_consume_targets",
        "hook_registry_after_consume_registrations",
        "hook_after_all_reduce_targets",
        "hook_registry_after_all_reduce_registrations",
        "hook_after_communication_targets",
        "hook_registry_after_communication_registrations",
        "hook_after_all_to_all_targets",
        "hook_registry_after_all_to_all_registrations",
    ]


def test_coverage_report_requires_reduce_scatter_hooks_for_sharded_distribution():
    case = br.RegressionCase("m", "fp8", "2gpu_dp_zero2", op_kind="dense")
    results = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "failed",
            "metrics": {
                "block_schema_summary": {
                    "hook_after_all_reduce_targets": 1,
                    "hook_after_reduce_scatter_targets": 0,
                },
                "buffer_plan_summary": {
                    "hook_registry_after_all_reduce_registrations": 1,
                    "hook_registry_after_reduce_scatter_registrations": 0,
                },
            },
        }
    }

    report = br.coverage_report(results)

    assert report["rows"][0]["hook_readiness_status"] == "missing"
    assert report["rows"][0]["missing_hook_counts"] == [
        "hook_after_reduce_scatter_targets",
        "hook_registry_after_reduce_scatter_registrations",
    ]


def test_coverage_report_requires_full_hook_structural_targets():
    case = br.RegressionCase("m", "fp8", "single_gpu")
    results = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "passed",
            "metrics": {
                "descriptor_summary": {
                    "lora_slices": 2,
                    "lora_schema_slot_slices": 2,
                    "lora_schema_target_slices": 1,
                },
                "block_schema_summary": {"hook_after_produce_targets": 1},
                "buffer_plan_summary": {"hook_registry_after_produce_registrations": 1},
            },
        }
    }

    report = br.coverage_report(results)

    assert report["rows"][0]["hook_readiness_status"] == "missing"
    assert report["rows"][0]["missing_hook_counts"] == ["lora_schema_target_slices"]


def test_coverage_report_marks_descriptor_requirements_present():
    case = br.RegressionCase("m", "fp8", "single_gpu")
    results = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "passed",
            "metrics": {
                "descriptor_summary": {
                    "matmul_fp8_forward_eligible_ops": 8,
                    "matmul_fp8_backward_eligible_ops": 8,
                }
            },
        }
    }

    report = br.coverage_report(results)

    assert report["rows"][0]["descriptor_requirement_status"] == "present"
    assert report["rows"][0]["missing_descriptor_counts"] == []


def test_artifact_schema_accepts_descriptor_summary():
    artifacts = br.RegressionArtifacts.from_dict(
        {
            "activation_snapshots": {"layer0": {"max_abs": 1.0}},
            "descriptor_summary": {
                "compiled_ops": 12,
                "fusion_candidate_starts": 2,
            },
            "block_schema_summary": {
                "block_schema_records": 2,
                "block_schema_expected_layers": 2,
                "block_schema_missing_layers": 0,
            },
            "buffer_plan_summary": {
                "schema_record_count": 2,
                "schema_activation_shape_savings_bytes": 1024,
            },
            "arena_summary": {
                "arena_fwd_stack_bytes": 2048,
            },
        }
    )

    assert artifacts.descriptor_summary == {
        "compiled_ops": 12,
        "fusion_candidate_starts": 2,
    }
    assert artifacts.block_schema_summary == {
        "block_schema_records": 2,
        "block_schema_expected_layers": 2,
        "block_schema_missing_layers": 0,
    }
    assert artifacts.buffer_plan_summary == {
        "schema_record_count": 2,
        "schema_activation_shape_savings_bytes": 1024,
    }
    assert artifacts.arena_summary == {
        "arena_fwd_stack_bytes": 2048,
    }


def test_load_results_ignores_report_artifacts(tmp_path):
    case = br.REGRESSION_MATRIX[0]
    br.write_results([br.RegressionResult(case=br.asdict(case), status="skipped")], tmp_path)
    (tmp_path / "north_star_coverage.json").write_text(json.dumps({"metric": "fp8_fp4_coverage"}))

    loaded = br.load_results(tmp_path)

    assert set(loaded) == {case.case_id}


def test_write_results_can_normalize_volatile_baseline_fields(tmp_path):
    case = br.REGRESSION_MATRIX[0]
    result = br.RegressionResult(case=br.asdict(case), status="skipped", started_at=123.0, duration_s=4.5)

    br.write_results([result], tmp_path, stable=True)

    data = json.loads((tmp_path / f"{case.case_id}.json").read_text())
    assert data["started_at"] == 0.0
    assert data["duration_s"] == 0.0


def test_command_for_case_uses_sft_cli_without_unsupported_flags():
    case = br.RegressionCase("m", "bf16", "single_gpu", config="dummy.yaml")

    cmd = br._command_for_case(case)

    assert cmd[:2] == ["surogate", "sft"]
    assert "--max-steps" not in cmd


def test_materialize_case_config_applies_runner_overrides(tmp_path, monkeypatch):
    source = tmp_path / "case.yaml"
    source.write_text("model: upstream/model\nmax_steps: 100\neval_steps: 25\ngpus: 1\nep_size: 4\n")
    monkeypatch.setattr(br, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("MODEL_PATH", "/models/local")
    case = br.RegressionCase(
        "m", "fp8", "2gpu_dp", storage="cpu_stream", config="case.yaml", env_model_path="MODEL_PATH"
    )

    materialized = br._materialize_case_config(case, steps=7, directory=tmp_path / "out")

    data = br.yaml.safe_load(materialized.read_text())
    assert data["model"] == "/models/local"
    assert data["max_steps"] == 7
    assert data["eval_steps"] == 0
    assert data["output_dir"] == str(tmp_path / "out" / "runs" / case.case_id)
    assert data["gpus"] == 2
    assert data["ep_size"] == 1
    assert data["cpu_training"] is True


def test_materialize_case_config_keeps_hf_model_without_local_override(tmp_path, monkeypatch):
    source = tmp_path / "case.yaml"
    source.write_text("model: org/model\nmax_steps: 100\n")
    monkeypatch.setattr(br, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    case = br.RegressionCase("m", "bf16", "single_gpu", config="case.yaml", env_model_path="MODEL_PATH")

    materialized = br._materialize_case_config(case, steps=3, directory=tmp_path / "out")

    data = br.yaml.safe_load(materialized.read_text())
    assert data["model"] == "org/model"
    assert data["max_steps"] == 3


def test_select_cases_accepts_case_id_and_model_filters():
    case = br.REGRESSION_MATRIX[0]

    by_id = br.select_cases([case.case_id])
    by_model = br.select_cases([case.model])

    assert by_id == (case,)
    assert all(row.model == case.model for row in by_model)
    assert len(by_model) > 1


def test_select_cases_rejects_unknown_filters():
    try:
        br.select_cases(["missing-model"])
    except SystemExit as exc:
        assert "missing-model" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_case_listing_exposes_case_ids():
    case = br.REGRESSION_MATRIX[0]

    rows = br.case_listing((case,))

    assert rows == [
        {
            "case_id": case.case_id,
            "model": case.model,
            "recipe": case.recipe,
            "distribution": case.distribution,
            "storage": case.storage,
            "op_kind": case.op_kind,
            "supported": case.supported,
            "env_model_path": case.env_model_path,
            "config": case.config,
        }
    ]


def test_main_list_cases_prints_json(capsys):
    rc = br.main(["--list-cases", "--case", br.REGRESSION_MATRIX[0].case_id])

    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["case_id"] for row in rows] == [br.REGRESSION_MATRIX[0].case_id]


def test_supported_first_month_cases_have_configs():
    missing = [case.case_id for case in br.REGRESSION_MATRIX if case.supported and not case.config]

    assert missing == []


def test_main_defaults_to_five_steps_and_timeout(tmp_path, monkeypatch):
    captured = []

    def fake_run_case(case, *, run, steps, artifact_dir, timeout_s):
        captured.append((case.case_id, run, steps, artifact_dir, timeout_s))
        return br.RegressionResult(case=br.asdict(case), status="skipped")

    monkeypatch.setattr(br, "run_case", fake_run_case)

    rc = br.main(["--case", br.REGRESSION_MATRIX[0].case_id, "--out", str(tmp_path)])

    assert rc == 0
    assert captured == [(br.REGRESSION_MATRIX[0].case_id, False, 5, tmp_path, br.DEFAULT_RUN_TIMEOUT_S)]


def test_main_accepts_timeout_override(tmp_path, monkeypatch):
    captured = []

    def fake_run_case(case, *, run, steps, artifact_dir, timeout_s):
        captured.append(timeout_s)
        return br.RegressionResult(case=br.asdict(case), status="skipped")

    monkeypatch.setattr(br, "run_case", fake_run_case)

    rc = br.main(["--case", br.REGRESSION_MATRIX[0].case_id, "--out", str(tmp_path), "--timeout-s", "7"])

    assert rc == 0
    assert captured == [7]


def test_compare_artifacts_respects_recipe_tolerance():
    case = br.RegressionCase("m", "fp8", "single_gpu")
    base = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "passed",
            "metrics": {"activation_snapshots": {"layer0": {"max_abs": 1.0}}},
        }
    }
    cur = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "passed",
            "metrics": {"activation_snapshots": {"layer0": {"max_abs": 1.04}}},
        }
    }

    assert br.compare_results(cur, base) == []


def test_compare_artifacts_fails_on_tolerance_breach():
    case = br.RegressionCase("m", "bf16", "single_gpu")
    base = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "passed",
            "metrics": {"gradient_snapshots": {"w": {"max_abs": 1.0}}},
        }
    }
    cur = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "passed",
            "metrics": {"gradient_snapshots": {"w": {"max_abs": 1.02}}},
        }
    }

    failures = br.compare_results(cur, base)

    assert failures == [f"{case.case_id}: gradient.w.max_abs delta 0.02 > 0.01"]


def test_compare_artifacts_checks_convergence_perf_memory_and_nccl():
    case = br.RegressionCase("m", "bf16", "2gpu_dp")
    base_metrics = {
        "convergence_curve": [{"step": 50, "loss": 1.0}],
        "step_time_ms": {"p50": 100.0},
        "cuda_peak_memory_bytes": 1000,
        "nccl": {"all_reduce": {"count": 2, "bytes": 1000}},
    }
    cur_metrics = {
        "convergence_curve": [{"step": 50, "loss": 1.2}],
        "step_time_ms": {"p50": 108.0},
        "cuda_peak_memory_bytes": 1200,
        "nccl": {"all_reduce": {"count": 3, "bytes": 1300}},
    }
    base = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": base_metrics}}
    cur = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": cur_metrics}}

    failures = br.compare_results(cur, base)

    assert any("convergence step 50" in f for f in failures)
    assert any("step_time_ms.p50 regression" in f for f in failures)
    assert any("cuda_peak_memory_bytes regression" in f for f in failures)
    assert any("nccl.all_reduce.count" in f for f in failures)
    assert any("nccl.all_reduce.bytes" in f for f in failures)


def test_compare_artifacts_checks_descriptor_summary_counts():
    case = br.RegressionCase("m", "fp8", "single_gpu")
    base_metrics = {"descriptor_summary": {"fusion_candidate_starts": 2, "forward_fp8_eligible_ops": 8}}
    cur_metrics = {"descriptor_summary": {"fusion_candidate_starts": 3}}
    base = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": base_metrics}}
    cur = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": cur_metrics}}

    failures = br.compare_results(cur, base)

    assert f"{case.case_id}: descriptor_summary.fusion_candidate_starts 3 != 2" in failures
    assert f"{case.case_id}: missing descriptor_summary.forward_fp8_eligible_ops" in failures


def test_compare_artifacts_checks_block_schema_summary_counts():
    case = br.RegressionCase("m", "bf16", "single_gpu")
    base_metrics = {"block_schema_summary": {"block_schema_records": 2, "block_schema_missing_layers": 0}}
    cur_metrics = {"block_schema_summary": {"block_schema_records": 1}}
    base = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": base_metrics}}
    cur = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": cur_metrics}}

    failures = br.compare_results(cur, base)

    assert f"{case.case_id}: block_schema_summary.block_schema_records 1 != 2" in failures
    assert f"{case.case_id}: missing block_schema_summary.block_schema_missing_layers" in failures


def test_compare_artifacts_checks_buffer_plan_summary_counts():
    case = br.RegressionCase("m", "bf16", "single_gpu")
    base_metrics = {
        "buffer_plan_summary": {
            "schema_record_count": 2,
            "schema_activation_shape_savings_bytes": 1024,
        }
    }
    cur_metrics = {"buffer_plan_summary": {"schema_record_count": 1}}
    base = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": base_metrics}}
    cur = {case.case_id: {"case": br.asdict(case), "status": "passed", "metrics": cur_metrics}}

    failures = br.compare_results(cur, base)

    assert f"{case.case_id}: buffer_plan_summary.schema_record_count 1 != 2" in failures
    assert f"{case.case_id}: missing buffer_plan_summary.schema_activation_shape_savings_bytes" in failures


def test_compare_artifacts_checks_arena_summary_counts():
    case = br.REGRESSION_MATRIX[0]
    base_metrics = {"arena_summary": {"arena_fwd_stack_bytes": 2048, "forward_region_fwdstack_tid_count": 3}}
    cur_metrics = {"arena_summary": {"arena_fwd_stack_bytes": 1024}}
    base = {"case": br.asdict(case), "status": "passed", "metrics": base_metrics}
    cur = {"case": br.asdict(case), "status": "passed", "metrics": cur_metrics}

    failures = br._compare_artifacts(case.case_id, cur, base)

    assert f"{case.case_id}: arena_summary.arena_fwd_stack_bytes 1024 != 2048" in failures
    assert f"{case.case_id}: missing arena_summary.forward_region_fwdstack_tid_count" in failures


def test_run_case_loads_external_artifact(tmp_path, monkeypatch):
    case = br.RegressionCase("m", "bf16", "single_gpu", config="dummy.yaml")
    artifact_payload = {
        "activation_snapshots": {"layer0": {"max_abs": 1.0}},
        "convergence_curve": [{"step": 1, "loss": 2.0}],
    }
    script = (
        "import json, os, pathlib; "
        "pathlib.Path(os.environ['SUROGATE_REGRESSION_ARTIFACT']).write_text("
        f"{json.dumps(json.dumps(artifact_payload))})"
    )
    monkeypatch.setattr(br, "_missing_reason", lambda _: None)
    monkeypatch.setattr(br, "_materialize_case_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(br, "_command_for_case", lambda _, config_path=None: [sys.executable, "-c", script])

    result = br.run_case(case, run=True, steps=1, artifact_dir=tmp_path)

    assert result.status == "passed"
    assert result.metrics["activation_snapshots"] == artifact_payload["activation_snapshots"]
    assert result.metrics["convergence_curve"] == artifact_payload["convergence_curve"]


def test_run_case_reports_timeout(tmp_path, monkeypatch):
    case = br.RegressionCase("m", "bf16", "single_gpu", config="dummy.yaml")
    script = "import time; print('started', flush=True); time.sleep(5)"
    monkeypatch.setattr(br, "_missing_reason", lambda _: None)
    monkeypatch.setattr(br, "_materialize_case_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(br, "_command_for_case", lambda _, config_path=None: [sys.executable, "-c", script])

    result = br.run_case(case, run=True, steps=1, artifact_dir=tmp_path, timeout_s=0.1)

    assert result.status == "failed"
    assert result.reason == "command timed out after 0.1s"
    assert result.artifacts["returncode"] == "timeout"
    assert "started" in result.artifacts["stdout_tail"]
