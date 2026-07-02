import json

import pytest

from ultra.config import WorkerSpec
from ultra.executor import execute_workflow
from ultra.failure_taxonomy import (
    build_failure_taxonomy_report,
    classify_rollout_outcome,
    reward_for_class,
)
from ultra.harness import HARNESS_REGISTRY, StepResult
from ultra.schemas import (
    EnvironmentSpec,
    ExecStep,
    Grade,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
    Workflow,
    WorkflowStep,
)
from ultra.workers import FakeProvider, Sampling, WorkerPool


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="taxonomy-task",
        capability="factual_qa",
        source=SourceRef(name="existing_bank", version="v1", policy="train_allowed"),
        input=TaskInput(messages=[{"role": "user", "content": "Q"}]),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="mc_letter", expected_answer="C"),
        splitting=SplittingSpec(group_id="g", split="grpo_train"),
    )


def _pool() -> WorkerPool:
    return WorkerPool([WorkerSpec(worker_id="w0", model="fake/model")], FakeProvider())


def test_failure_taxonomy_classifies_core_outcomes():
    assert reward_for_class("invalid_workflow_trainable") == 0.0
    assert reward_for_class("valid_incorrect_trainable") == 0.5
    assert reward_for_class("valid_correct_trainable") == 1.0
    assert reward_for_class("provider_failure_retry_or_exclude") is None

    assert (
        classify_rollout_outcome(
            workflow_parse_valid=False,
            grade=None,
            exec_steps=[],
            failure_class="invalid_workflow_trainable: bad worker",
        )
        == "invalid_workflow_trainable"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=1.0, success=True),
            exec_steps=[ExecStep(worker_id=0, harness="direct_qa")],
        )
        == "valid_correct_trainable"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False),
            exec_steps=[ExecStep(worker_id=0, harness="direct_qa")],
        )
        == "valid_incorrect_trainable"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False),
            exec_steps=[ExecStep(worker_id=0, harness="direct_qa", termination="timeout")],
        )
        == "budget_exhausted_trainable"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False, details={"error": "no Harbor verifier rewards found"}),
            exec_steps=[ExecStep(worker_id=0, harness="terminal_sandbox", termination="harbor_timeout")],
        )
        == "budget_exhausted_trainable"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False),
            exec_steps=[ExecStep(worker_id=0, harness="direct_qa", termination="missing_provider_key")],
        )
        == "provider_failure_retry_or_exclude"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=None,
            exec_steps=[],
            failure_class="harness_crash_exclude: RateLimitError: Error code: 429 - upstream saturated",
        )
        == "provider_failure_retry_or_exclude"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False),
            exec_steps=[ExecStep(worker_id=0, harness="long_context", termination="missing_context")],
        )
        == "task_setup_failure_quarantine"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False, details={"error": "no Harbor verifier rewards found"}),
            exec_steps=[ExecStep(worker_id=0, harness="terminal_sandbox")],
        )
        == "grader_crash_quarantine"
    )
    assert (
        classify_rollout_outcome(
            workflow_parse_valid=True,
            grade=Grade(score=0.0, success=False, details={"error": "harbor environment setup failed"}),
            exec_steps=[ExecStep(worker_id=0, harness="terminal_sandbox")],
        )
        == "harness_crash_exclude"
    )


@pytest.mark.asyncio
async def test_executor_marks_infra_failure_non_training(monkeypatch):
    class ProviderFailureHarness:
        async def run_step(self, step, pool, sampling):
            return StepResult(text="", error="missing provider key", termination="missing_provider_key")

        def grade(self, task, final):
            return Grade(score=0.0, success=False)

    monkeypatch.setitem(HARNESS_REGISTRY, "direct_qa", ProviderFailureHarness)

    rec = await execute_workflow(
        _task(),
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve")]),
        _pool(),
        Sampling(),
        "provider-fail",
    )

    assert rec.outcome_class == "provider_failure_retry_or_exclude"
    assert rec.reward is None
    assert rec.valid_for_training is False
    assert rec.failure_class and rec.failure_class.startswith("provider_failure_retry_or_exclude")


@pytest.mark.asyncio
async def test_executor_marks_provider_quota_exception_non_training(monkeypatch):
    class ProviderQuotaHarness:
        async def run_step(self, step, pool, sampling):
            raise PermissionError("Error code: 403 - local:insufficient_quota")

    monkeypatch.setitem(HARNESS_REGISTRY, "direct_qa", ProviderQuotaHarness)

    rec = await execute_workflow(
        _task(),
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve")]),
        _pool(),
        Sampling(),
        "provider-quota",
    )

    assert rec.outcome_class == "provider_failure_retry_or_exclude"
    assert rec.reward is None
    assert rec.valid_for_training is False
    assert rec.failure_class and "insufficient_quota" in rec.failure_class


@pytest.mark.asyncio
async def test_executor_marks_grader_exception_quarantine(monkeypatch):
    class GraderCrashHarness:
        async def run_step(self, step, pool, sampling):
            return StepResult(text="Answer: C")

        def grade(self, task, final):
            raise RuntimeError("grader died")

    monkeypatch.setitem(HARNESS_REGISTRY, "direct_qa", GraderCrashHarness)

    rec = await execute_workflow(
        _task(),
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve")]),
        _pool(),
        Sampling(),
        "grader-crash",
    )

    assert rec.outcome_class == "grader_crash_quarantine"
    assert rec.reward is None
    assert rec.valid_for_training is False
    assert rec.failure_class and "grader died" in rec.failure_class


def test_failure_taxonomy_report_writes_artifact(tmp_path):
    report = build_failure_taxonomy_report(
        manifest_dir=tmp_path / "manifest",
        report_out=tmp_path / "failure_taxonomy_report.json",
        md_out=tmp_path / "failure_taxonomy_report.md",
        created_at_utc="2026-06-27T00:00:00Z",
    )

    assert report["frozen"] is True
    assert report["taxonomy_sha256"].startswith("sha256:")
    assert report["entries"]["valid_incorrect_trainable"]["reward"] == 0.5
    assert "missing_provider_key" in report["termination_mapping"]["provider_failure_retry_or_exclude"]
    assert json.loads((tmp_path / "failure_taxonomy_report.json").read_text())["version"]
    assert "Failure Taxonomy" in (tmp_path / "failure_taxonomy_report.md").read_text()
