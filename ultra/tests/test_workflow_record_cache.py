from ultra.schemas import (
    ConductorRecord,
    EnvironmentSpec,
    Execution,
    Grade,
    GraderSpec,
    RolloutRecord,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
    Workflow,
    WorkflowStep,
)
from ultra.workflow_record_cache import WorkflowRecordCache, workflow_record_cache_key
from ultra.workers import Sampling


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="task-1",
        capability="general_reasoning",
        source=SourceRef(name="unit", version="1", policy="train_allowed"),
        input=TaskInput(messages=[{"role": "user", "content": "answer this"}]),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="exact", expected_answer="ok"),
        splitting=SplittingSpec(group_id="g1", split="grpo_train"),
    )


def _workflow() -> Workflow:
    return Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve", access=[], budget="short")])


def test_workflow_record_cache_key_includes_worker_model():
    base = {
        "task": _task(),
        "workflow": _workflow(),
        "worker_ids": ["direct:gpt"],
        "worker_harnesses": {"direct:gpt": "direct_qa"},
        "sampling": Sampling(temperature=0.2, max_tokens=8192, seed=7, reasoning_effort="high"),
        "max_steps": 3,
    }

    key_a = workflow_record_cache_key(worker_models={"direct:gpt": "gpt-5.5"}, **base)
    key_b = workflow_record_cache_key(worker_models={"direct:gpt": "gpt-5.6"}, **base)

    assert key_a != key_b


def test_workflow_record_cache_round_trips_rollout_record(tmp_path):
    cache = WorkflowRecordCache(tmp_path)
    record = RolloutRecord(
        rollout_id="r1",
        task_id="task-1",
        source_name="unit",
        capability="general_reasoning",
        harness="direct_qa",
        conductor=ConductorRecord(raw_output='{"steps":[]}'),
        workflow=_workflow(),
        execution=Execution(),
        grade=Grade(score=1.0, success=True),
        reward=1.0,
        outcome_class="valid_correct_trainable",
        valid_for_training=True,
    )

    cache.set("abc123", record)
    loaded = cache.get("abc123")

    assert loaded is not None
    assert loaded.rollout_id == "r1"
    assert loaded.reward == 1.0
    assert loaded.grade and loaded.grade.success is True

