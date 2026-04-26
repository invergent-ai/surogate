import json

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
