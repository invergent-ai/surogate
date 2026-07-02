import json
from types import SimpleNamespace

import pytest

from ultra.harness import HARNESS_REGISTRY, StepResult
from ultra.scaffold_canary import run_canary, run_cli, select_arm, select_task
from ultra.schemas import EnvironmentSpec, Grade, GraderSpec, SourceRef, SplittingSpec, TaskInput, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="repo-canary",
        capability="agentic_coding",
        source=SourceRef(name="s", version="v", policy="pool_only"),
        input=TaskInput(messages=[{"role": "user", "content": "Fix it"}]),
        environment=EnvironmentSpec(harness="opencode"),
        grader=GraderSpec(type="hidden_tests"),
        splitting=SplittingSpec(group_id="g", split="pool_validation"),
    )


def test_select_task_and_arm():
    task = _task()
    assert select_task([task], "repo-canary") is task
    assert select_arm("solo__codex_gpt_coding_agent").name == "solo__codex_gpt_coding_agent"


@pytest.mark.asyncio
async def test_scaffold_canary_runs_one_arm_with_fake_harness(monkeypatch, tmp_path):
    class FakeHarness:
        name = "codex"

        async def run_step(self, step, pool, sampling):
            assert step.worker_id == "codex_gpt_coding_agent"
            return StepResult(text=f"{step.worker_id}:{step.subtask}")

        def grade(self, task, final):
            return Grade(score=1.0, success=True)

    monkeypatch.setitem(HARNESS_REGISTRY, "codex", FakeHarness)

    tasks_jsonl = tmp_path / "tasks.jsonl"
    tasks_jsonl.write_text(json.dumps(_task().model_dump(mode="json")) + "\n")

    record = await run_canary(
        tasks_jsonl=tasks_jsonl,
        task_id="repo-canary",
        arm_name="solo__codex_gpt_coding_agent",
        rollout_id="canary-test",
        budget="short",
    )

    assert record.rollout_id == "canary-test"
    assert record.reward == 1.0
    assert record.execution.steps[0].harness == "codex"
    assert record.execution.steps[0].budget == "short"
    assert record.workflow.steps[0].budget == "short"


@pytest.mark.asyncio
async def test_scaffold_canary_cli_writes_artifacts_and_agent_trace(monkeypatch, tmp_path):
    class FakeHarness:
        name = "codex"

        async def run_step(self, step, pool, sampling):
            artifact_dir = tmp_path / "artifacts" / "step"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            patch = artifact_dir / "patch.diff"
            prompt = artifact_dir / "prompt.txt"
            command = artifact_dir / "command.json"
            workspace = artifact_dir / "workspace_snapshot"
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

    monkeypatch.setitem(HARNESS_REGISTRY, "codex", FakeHarness)
    tasks_jsonl = tmp_path / "tasks.jsonl"
    tasks_jsonl.write_text(json.dumps(_task().model_dump(mode="json")) + "\n")
    out = tmp_path / "rollout.json"
    trace_out = tmp_path / "trace.json"

    await run_cli(
        SimpleNamespace(
            tasks_jsonl=str(tasks_jsonl),
            task_id="repo-canary",
            arm="solo__codex_gpt_coding_agent",
            rollout_id="trace-cli-test",
            temperature=0.2,
            max_tokens=4096,
            reasoning="high",
            budget="short",
            out=str(out),
            artifact_dir=str(tmp_path / "artifacts"),
            agent_trace_out=str(trace_out),
        )
    )

    assert out.exists()
    assert trace_out.exists()
    trace = json.loads(trace_out.read_text())
    assert trace["trace_id"] == "trace-cli-test"
    assert trace["origin_harness"] == "codex"
    assert trace["artifacts"]["final_patch_ref"]
    assert trace["artifacts"]["workspace_snapshot_ref"]
    assert trace["artifacts"]["hidden_grade_ref"]
