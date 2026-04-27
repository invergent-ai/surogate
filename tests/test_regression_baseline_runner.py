import json
import sys

from surogate.regression import baseline_runner as br


def test_runner_records_skips_without_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN3_MODEL_PATH", raising=False)

    result = br.run_case(br.FIRST_MONTH_MATRIX[0], run=False, steps=1)

    assert result.status == "skipped"
    assert "missing QWEN3_MODEL_PATH" in result.reason


def test_compare_detects_status_drift():
    case = br.FIRST_MONTH_MATRIX[0]
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
    assert json.dumps(report)


def test_coverage_report_marks_moe_grouped_capabilities():
    case = br.RegressionCase("m", "fp8", "single_gpu", op_kind="moe_grouped")
    results = {
        case.case_id: {
            "case": br.asdict(case),
            "status": "skipped",
            "reason": "missing weights",
            "metrics": {
                "descriptor_summary": {
                    "fusion_candidate_starts": 3,
                    "forward_matmul_fp8_forward_eligible_ops": 4,
                    "backward_matmul_fp8_backward_eligible_ops": 4,
                }
            },
        }
    }

    report = br.coverage_report(results)

    assert report["rows"][0]["required_capabilities"] == ["GroupedMatmul", "MoERouted", "FP8Eligible"]
    assert report["rows"][0]["required_moe_capabilities"] == ["GroupedGemmEligible", "FP8GroupedEligible"]
    assert report["rows"][0]["required_matmul_capabilities"] == []
    assert report["rows"][0]["matmul_capability_counts"]["forward_matmul_fp8_forward_eligible_ops"] == 4
    assert report["rows"][0]["matmul_capability_counts"]["backward_matmul_fp8_backward_eligible_ops"] == 4
    assert report["rows"][0]["fusion_candidate_starts"] == 3


def test_artifact_schema_accepts_descriptor_summary():
    artifacts = br.RegressionArtifacts.from_dict(
        {
            "activation_snapshots": {"layer0": {"max_abs": 1.0}},
            "descriptor_summary": {
                "compiled_ops": 12,
                "fusion_candidate_starts": 2,
            },
        }
    )

    assert artifacts.descriptor_summary == {
        "compiled_ops": 12,
        "fusion_candidate_starts": 2,
    }


def test_load_results_ignores_report_artifacts(tmp_path):
    case = br.FIRST_MONTH_MATRIX[0]
    br.write_results([br.RegressionResult(case=br.asdict(case), status="skipped")], tmp_path)
    (tmp_path / "north_star_coverage.json").write_text(json.dumps({"metric": "fp8_fp4_coverage"}))

    loaded = br.load_results(tmp_path)

    assert set(loaded) == {case.case_id}


def test_write_results_can_normalize_volatile_baseline_fields(tmp_path):
    case = br.FIRST_MONTH_MATRIX[0]
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


def test_select_cases_accepts_case_id_and_model_filters():
    case = br.FIRST_MONTH_MATRIX[0]

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
    case = br.FIRST_MONTH_MATRIX[0]

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
    rc = br.main(["--list-cases", "--case", br.FIRST_MONTH_MATRIX[0].case_id])

    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["case_id"] for row in rows] == [br.FIRST_MONTH_MATRIX[0].case_id]


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
