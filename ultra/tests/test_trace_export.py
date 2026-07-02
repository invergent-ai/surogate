from pathlib import Path

from ultra.schemas import (
    EnvironmentSpec,
    Execution,
    ExecStep,
    Grade,
    GraderSpec,
    RepoRef,
    RolloutRecord,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
    Workflow,
    WorkflowStep,
)
from ultra.trace_export import rollout_to_agent_trace


def test_rollout_to_agent_trace_preserves_branch_required_refs():
    task = TaskSpec(
        task_id="generated_repo_tasks__x",
        capability="agentic_coding",
        source=SourceRef(name="generated_repo_tasks", version="v1", policy="train_allowed"),
        input=TaskInput(
            messages=[{"role": "user", "content": "Fix x"}],
            repo=RepoRef(url="local://generated_repo_tasks/x", base_commit="generated-v1"),
        ),
        environment=EnvironmentSpec(harness="opencode"),
        grader=GraderSpec(type="deep_swe_hidden_tests"),
        splitting=SplittingSpec(group_id="g", split="grpo_train"),
    )
    rollout = RolloutRecord(
        rollout_id="r1",
        task_id=task.task_id,
        source_name=task.source.name,
        capability=task.capability,
        harness="opencode",
        workflow=Workflow(steps=[WorkflowStep(worker_id=2, subtask="fix")]),
        execution=Execution(
            steps=[
                ExecStep(
                    worker_id=2,
                    harness="opencode",
                    session_ref=str(Path("/tmp/workspace")),
                    patch_ref=str(Path("/tmp/patch.diff")),
                    messages_ref=str(Path("/tmp/prompt.txt")),
                    tool_events_ref=str(Path("/tmp/command.json")),
                    text="diff --git a/x b/x\n",
                )
            ]
        ),
        grade=Grade(score=1.0, success=True, grader_ref=str(Path("/tmp/grade.json"))),
        reward=1.0,
    )

    trace = rollout_to_agent_trace(rollout, task, worker_models={2: "moonshotai/kimi-k2.7-code"})

    assert trace.origin_harness == "opencode"
    assert trace.worker_model == "moonshotai/kimi-k2.7-code"
    assert trace.repo.url == "local://generated_repo_tasks/x"
    assert trace.repo.base_commit == "generated-v1"
    assert trace.artifacts.final_patch_ref == "/tmp/patch.diff"
    assert trace.artifacts.workspace_snapshot_ref == "/tmp/workspace"
    assert trace.artifacts.public_test_log_ref == "/tmp/command.json"
    assert trace.artifacts.hidden_grade_ref == "/tmp/grade.json"
    assert {event.type for event in trace.events} == {"message", "command", "file_edit", "test_result"}
