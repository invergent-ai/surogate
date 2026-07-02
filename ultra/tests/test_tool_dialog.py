import json

import pytest

from ultra.config import WorkerSpec
from ultra.executor import execute_workflow
from ultra.harness import HARNESS_REGISTRY
from ultra.harness.tool_dialog import ToolDialogHarness
from ultra.registry import TaskRegistry
from ultra.schemas import SourceManifest, Workflow, WorkflowStep
from ultra.tool_dialog_tasks import TASKS, materialize_tool_dialog_tasks, task_spec
from ultra.workers import FakeProvider, Sampling, ToolCall, ToolCompletion, WorkerPool


def test_tool_dialog_tasks_materialize_and_ingest(tmp_path):
    out = tmp_path / "tasks.jsonl"
    report = materialize_tool_dialog_tasks(out_jsonl=out, report_out=tmp_path / "report.json")

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = [task_spec(task) for task in TASKS]
    hard_rows = [row for row in rows if row["task_id"].split("__")[-1].startswith("hard-")]
    assert report["task_count"] == len(TASKS)
    assert len(rows) == len(TASKS)
    assert len(hard_rows) >= 6
    assert all("hard" in row["metadata"]["tags"] for row in hard_rows)
    assert all(len(row["grader"]["expected_answer"]["success"]) >= 3 for row in hard_rows)
    assert any(
        set(row["grader"]["expected_answer"]["allowed_tools"]) >= {"cancel_order", "update_shipping_address", "finish"}
        for row in hard_rows
    )
    assert {row["source"]["name"] for row in rows} == {"tau_custom"}
    assert {row["environment"]["harness"] for row in rows} == {"tool_dialog"}
    assert {row["splitting"]["split"] for row in rows} == {"grpo_train"}

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name="tau_custom",
            source_type="generated_tool_dialogue",
            version="v1",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(TASKS)


def test_tool_dialog_harness_registered():
    assert HARNESS_REGISTRY["tool_dialog"] is ToolDialogHarness


@pytest.mark.asyncio
async def test_tool_dialog_harness_executes_tool_call_and_grades_success():
    spec = task_spec(TASKS[0])
    calls = {"n": 0}

    def tool_fn(model, messages, tools, sampling):
        calls["n"] += 1
        if calls["n"] == 1:
            return ToolCompletion(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="cancel_order", arguments={"order_id": "O-100"}),
                    ToolCall(id="call_2", name="finish", arguments={}),
                ],
                model=model,
            )
        return ToolCompletion(content="done", tool_calls=[], model=model)

    pool = WorkerPool(
        [WorkerSpec(worker_id="tool_worker", model="fake/tool")],
        FakeProvider(tool_fn=tool_fn),
    )
    rec = await execute_workflow(
        spec,
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="complete the user request", access=[])]),
        pool,
        Sampling(),
        "tool-dialog-ok",
    )

    assert rec.grade and rec.grade.success is True
    assert rec.reward == 1.0
    assert rec.execution.steps[0].harness == "tool_dialog"
    assert json.loads(rec.execution.steps[0].text)["state"]["orders"]["O-100"]["status"] == "cancelled"
