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
    assert json.dumps(report)


def test_load_results_ignores_report_artifacts(tmp_path):
    case = br.FIRST_MONTH_MATRIX[0]
    br.write_results([br.RegressionResult(case=br.asdict(case), status="skipped")], tmp_path)
    (tmp_path / "north_star_coverage.json").write_text(json.dumps({"metric": "fp8_fp4_coverage"}))

    loaded = br.load_results(tmp_path)

    assert set(loaded) == {case.case_id}


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
    monkeypatch.setattr(br, "_command_for_case", lambda _, steps: [sys.executable, "-c", script])

    result = br.run_case(case, run=True, steps=1, artifact_dir=tmp_path)

    assert result.status == "passed"
    assert result.metrics["activation_snapshots"] == artifact_payload["activation_snapshots"]
    assert result.metrics["convergence_curve"] == artifact_payload["convergence_curve"]
