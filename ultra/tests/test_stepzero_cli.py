"""Offline tests for the step-zero entrypoint helpers (the live run needs OpenRouter)."""

import pytest

from ultra.stepzero import HeadroomReport
from ultra.stepzero_run import format_report, parse_workers, sample_bank_tasks


def test_parse_workers():
    ws = parse_workers("flash=deepseek/v4-flash, glm=z-ai/glm-5.2")
    assert [w.worker_id for w in ws] == ["flash", "glm"]
    assert ws[0].model == "deepseek/v4-flash"
    with pytest.raises(ValueError):
        parse_workers("noequals")
    with pytest.raises(ValueError):
        parse_workers("")


def _report(delta_cv, ci, oracle_signal) -> HeadroomReport:
    return HeadroomReport(
        n_tasks=100,
        n_reps=3,
        best_single=0.6,
        single_acc={0: 0.6},
        scaffold_acc={"A_direct": 0.6, "C_solve_critique_revise": 0.6 + max(delta_cv, 0)},
        best_scaffold=("C_solve_critique_revise", 0.6 + max(delta_cv, 0)),
        delta_fixed_cv=delta_cv,
        delta_fixed_ci=ci,
        oracle_obs=0.7,
        oracle_null=0.7 - oracle_signal,
        oracle_signal=oracle_signal,
    )


def test_format_report_verdicts():
    go = format_report(_report(0.05, (0.02, 0.08), 0.1), ["w0"], "direct_qa")
    assert "GO:" in go and "Δ_fixed" in go and "perm-null" in go
    maybe = format_report(_report(0.0, (-0.02, 0.02), 0.07), ["w0"], "direct_qa")
    assert "MAYBE" in maybe
    nogo = format_report(_report(0.0, (-0.03, 0.03), 0.01), ["w0"], "direct_qa")
    assert "NO-GO" in nogo


def test_sample_bank_tasks_filters():
    _adapter, tasks = sample_bank_tasks(
        20, split="grpo_train", harness="direct_qa", discriminative=False, seed=1
    )
    if not tasks:
        pytest.skip("bank not present or no matching tasks")
    assert len(tasks) <= 20
    for t in tasks:
        assert t.environment.harness == "direct_qa"
        assert t.splitting.split == "grpo_train"
