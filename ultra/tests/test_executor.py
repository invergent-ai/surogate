"""Multi-step executor end-to-end, offline (FakeProvider)."""

import pytest

from ultra.config import WorkerSpec
from ultra.executor import execute_workflow
from ultra.schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
    Workflow,
    WorkflowStep,
)
from ultra.workers import FakeProvider, Sampling, WorkerPool


def _mc_task(sol="C") -> TaskSpec:
    return TaskSpec(
        task_id="t",
        capability="factual_qa",
        source=SourceRef(name="s", version="v", policy="train_allowed"),
        input=TaskInput(messages=[{"role": "system", "content": "Answer: X"}, {"role": "user", "content": "Q"}]),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="mc_letter", expected_answer=sol),
        splitting=SplittingSpec(group_id="g", split="grpo_train"),
    )


def _ctx_aware_pool(n_workers=2) -> WorkerPool:
    # Returns the right answer only once it can see a prior step's result — so a passing
    # final step proves the executor flowed context along the access list.
    def fn(model, messages, sampling):
        joined = " ".join(m["content"] for m in messages)
        return "Answer: C" if "Authorized prior-step results" in joined else "Answer: A"

    return WorkerPool(
        [WorkerSpec(worker_id=f"w{i}", model="fake/model") for i in range(n_workers)],
        FakeProvider(fn),
    )


@pytest.mark.asyncio
async def test_multistep_flows_context_along_access():
    pool = _ctx_aware_pool(2)
    wf = Workflow(
        steps=[
            WorkflowStep(worker_id=0, subtask="plan", access=[]),
            WorkflowStep(worker_id=1, subtask="solve using the plan", access=[0]),
        ]
    )
    rec = await execute_workflow(_mc_task("C"), wf, pool, Sampling(), "r1")
    assert len(rec.execution.steps) == 2
    assert rec.execution.steps[0].text == "Answer: A"  # no prior context
    assert rec.execution.steps[1].text == "Answer: C"  # saw the prior step
    assert rec.grade.success is True
    assert rec.reward == 1.0
    assert rec.execution.steps[1].worker_id == 1


@pytest.mark.asyncio
async def test_invalid_workflow_short_circuits_to_zero():
    pool = _ctx_aware_pool(2)
    bad = Workflow(steps=[WorkflowStep(worker_id=9, subtask="x", access=[])])  # worker out of range
    rec = await execute_workflow(_mc_task(), bad, pool, Sampling(), "r2")
    assert rec.reward == 0.0
    assert rec.outcome_class == "invalid_workflow_trainable"
    assert rec.grade is None
    assert rec.conductor.workflow_parse_valid is False
    assert rec.failure_class and "invalid_workflow" in rec.failure_class
    assert rec.execution.steps == []
    assert rec.valid_for_training is True  # malformed workflows still train (reward 0)


@pytest.mark.asyncio
async def test_executor_routes_each_step_to_worker_scaffold_harness(monkeypatch):
    from ultra.harness import HARNESS_REGISTRY, StepResult
    from ultra.schemas import Grade

    def harness_cls(name):
        class _Harness:
            async def run_step(self, step, pool, sampling):
                prior_harnesses = ",".join(str(a["harness"]) for a in step.prior_artifacts)
                return StepResult(
                    text=(
                        f"{name}|step={step.step_index}|worker={step.worker_id}|"
                        f"access={step.access}|budget={step.budget}|prior={prior_harnesses}"
                    )
                )

            def grade(self, task, final):
                return Grade(score=1.0, success=final.text.startswith("opencode|step=2"))

        return _Harness

    monkeypatch.setitem(HARNESS_REGISTRY, "codex", harness_cls("codex"))
    monkeypatch.setitem(HARNESS_REGISTRY, "claude_code", harness_cls("claude_code"))
    monkeypatch.setitem(HARNESS_REGISTRY, "opencode", harness_cls("opencode"))

    pool = WorkerPool(
        [
            WorkerSpec(worker_id="codex_gpt_coding_agent", model="fake/codex"),
            WorkerSpec(worker_id="claude_code_opus_debugger", model="fake/claude"),
            WorkerSpec(worker_id="opencode_kimi_builder", model="fake/opencode"),
        ],
        FakeProvider(),
    )
    wf = Workflow(
        steps=[
            WorkflowStep(worker_id=0, subtask="plan/build", access=[]),
            WorkflowStep(worker_id=1, subtask="debug", access=[0], budget="short"),
            WorkflowStep(worker_id=2, subtask="repair", access=[0, 1], budget="long"),
        ]
    )
    rec = await execute_workflow(
        _mc_task("C"),
        wf,
        pool,
        Sampling(),
        "mixed-scaffold",
        worker_harnesses={
            "codex_gpt_coding_agent": "codex",
            "claude_code_opus_debugger": "claude_code",
            "opencode_kimi_builder": "opencode",
        },
    )

    assert [step.harness for step in rec.execution.steps] == ["codex", "claude_code", "opencode"]
    assert [step.budget for step in rec.execution.steps] == ["medium", "short", "long"]
    assert "budget=long" in rec.execution.steps[2].text
    assert "prior=codex,claude_code" in rec.execution.steps[2].text
    assert rec.grade.success is True
    assert rec.reward == 1.0
    assert rec.outcome_class == "valid_correct_trainable"


@pytest.mark.asyncio
async def test_executor_classifies_provider_exception_separately(monkeypatch):
    from ultra.harness import HARNESS_REGISTRY
    from ultra.schemas import Grade

    class ProviderFailHarness:
        async def run_step(self, step, pool, sampling):
            raise RuntimeError("RateLimitError: Error code: 429 - upstream saturated")

        def grade(self, task, final):
            return Grade(score=0.0, success=False)

    monkeypatch.setitem(HARNESS_REGISTRY, "direct_qa", ProviderFailHarness)
    pool = WorkerPool([WorkerSpec(worker_id="w0", model="fake/model")], FakeProvider())
    rec = await execute_workflow(
        _mc_task("C"),
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve")]),
        pool,
        Sampling(),
        "provider-fail",
    )

    assert rec.outcome_class == "provider_failure_retry_or_exclude"
    assert rec.reward is None
    assert rec.valid_for_training is False
    assert rec.failure_class and rec.failure_class.startswith("provider_failure_retry_or_exclude")


@pytest.mark.asyncio
async def test_executor_records_artifact_refs_and_grade_ref(monkeypatch, tmp_path):
    from pathlib import Path

    from ultra.harness import HARNESS_REGISTRY, StepResult
    from ultra.schemas import Grade

    class ArtifactHarness:
        async def run_step(self, step, pool, sampling):
            step_dir = Path(step.artifact_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            patch = step_dir / "patch.diff"
            prompt = step_dir / "prompt.txt"
            command = step_dir / "command.json"
            workspace = step_dir / "workspace_snapshot"
            patch.write_text("diff --git a/x b/x\n")
            prompt.write_text("Fix it")
            command.write_text("{}")
            workspace.mkdir()
            return StepResult(
                text=patch.read_text(),
                patch_ref=str(patch),
                messages_ref=str(prompt),
                tool_events_ref=str(command),
                workspace_snapshot_ref=str(workspace),
            )

        def grade(self, task, final):
            return Grade(score=1.0, success=True)

    monkeypatch.setitem(HARNESS_REGISTRY, "opencode", ArtifactHarness)
    pool = WorkerPool([WorkerSpec(worker_id="w0", model="fake/model")], FakeProvider())
    rec = await execute_workflow(
        _mc_task("C"),
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="fix")]),
        pool,
        Sampling(),
        "artifact-rollout",
        worker_harnesses={"w0": "opencode"},
        artifact_dir=tmp_path / "artifacts",
    )

    step = rec.execution.steps[0]
    assert step.patch_ref and Path(step.patch_ref).exists()
    assert step.messages_ref and Path(step.messages_ref).exists()
    assert step.tool_events_ref and Path(step.tool_events_ref).exists()
    assert step.session_ref and Path(step.session_ref).exists()
    assert rec.grade and rec.grade.grader_ref
    assert Path(rec.grade.grader_ref).exists()
