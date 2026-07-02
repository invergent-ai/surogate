import pytest
from pydantic import ValidationError

from ultra.schemas import (
    AgentTrace,
    EnvironmentSpec,
    GraderSpec,
    RolloutRecord,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
    WorkerIdentity,
    Workflow,
    WorkflowStep,
)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="t1",
        capability="math",
        source=SourceRef(name="s", version="v", policy="train_allowed"),
        input=TaskInput(messages=[{"role": "user", "content": "2+2?"}]),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="math_equal", expected_answer="4"),
        splitting=SplittingSpec(group_id="g", split="grpo_train"),
    )


def test_taskspec_roundtrips():
    t = _task()
    assert TaskSpec(**t.model_dump()) == t
    assert t.schema_version == "2.0"


def test_bad_harness_rejected():
    with pytest.raises(ValidationError):
        EnvironmentSpec(harness="nope")


def test_bad_policy_rejected():
    with pytest.raises(ValidationError):
        SourceRef(name="s", version="v", policy="whatever")


def test_rollout_record_minimal():
    rec = RolloutRecord(
        rollout_id="r1",
        task_id="t1",
        source_name="s",
        capability="math",
        harness="direct_qa",
        workflow=Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve")]),
        reward=1.0,
    )
    assert rec.workflow.steps[0].access == []
    assert rec.workflow.steps[0].budget == "medium"
    assert rec.execution.steps == []
    assert rec.valid_for_training is True


def test_workflow_step_budget_validation():
    assert WorkflowStep(worker_id=0, subtask="probe", budget="short").budget == "short"
    with pytest.raises(ValidationError):
        WorkflowStep(worker_id=0, subtask="probe", budget="forever")


def test_agent_trace_and_worker_identity_roundtrip():
    trace = AgentTrace(
        trace_id="trace_1",
        origin_harness="codex",
        harness_version="1.0",
        worker_model="gpt-5.5",
        task_id="swe__1",
        prompt={"user_task": "Fix the bug"},
        events=[{"type": "command", "agent_turn": 1, "content_ref": "artifact://cmd"}],
        grade={"score": 1.0, "success": True},
        usage={"cost_usd": None, "wall_time_seconds": 12.5},
        privacy={"redacted": True, "contains_user_secret": False, "license_status": "ok_for_internal_training"},
    )
    assert AgentTrace(**trace.model_dump()) == trace
    assert trace.usage.cost_usd is None

    worker = WorkerIdentity(
        worker_id=0,
        name="codex_gpt_coding_agent",
        backend="codex",
        model="gpt-5.5",
        role_prior=["builder", "repair"],
        tool_permissions={"read_files": True, "edit_files": True, "run_tests": True},
    )
    assert worker.tool_permissions.run_tests is True
