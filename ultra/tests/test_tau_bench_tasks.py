import json
import sys
import types

import pytest

from ultra.config import WorkerSpec
from ultra.executor import execute_workflow
from ultra.harness import HARNESS_REGISTRY
from ultra.harness.tau_bench import TauBenchHarness, _max_turns_for_step
from ultra.registry import TaskRegistry
from ultra.schemas import SourceManifest, Workflow, WorkflowStep
from ultra.tau_bench_tasks import (
    DEFAULT_RETAIL_TRAIN_INDICES,
    high_action_retail_train_indices,
    materialize_tau_bench_tasks,
    task_spec,
    TauBenchTask,
)
from ultra.workers import FakeProvider, Sampling, ToolCall, ToolCompletion, WorkerPool


def test_tau_bench_tasks_materialize_and_ingest(tmp_path):
    out = tmp_path / "tasks.jsonl"
    report = materialize_tau_bench_tasks(out_jsonl=out, report_out=tmp_path / "report.json", limit=3)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert report["task_count"] == 3
    assert report["task_indices"] == list(DEFAULT_RETAIL_TRAIN_INDICES[:3])
    assert len(rows) == 3
    assert {row["source"]["name"] for row in rows} == {"tau_bench_retail_train"}
    assert {row["environment"]["harness"] for row in rows} == {"tau_bench"}
    assert {row["splitting"]["split"] for row in rows} == {"grpo_train"}
    assert all(row["grader"]["expected_answer"]["task_split"] == "train" for row in rows)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name="tau_bench_retail_train",
            source_type="tau_bench",
            version="sierra-59a200c6",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many([task_spec(TauBenchTask(env_name="retail", task_split="train", task_index=i)) for i in DEFAULT_RETAIL_TRAIN_INDICES[:3]])
    assert len(registry) == 3


def test_tau_bench_tasks_support_offset(tmp_path):
    out = tmp_path / "tail.jsonl"
    report = materialize_tau_bench_tasks(out_jsonl=out, limit=2, offset=6)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert report["offset"] == 6
    assert report["task_indices"] == list(DEFAULT_RETAIL_TRAIN_INDICES[6:8])
    assert [row["grader"]["expected_answer"]["task_index"] for row in rows] == list(DEFAULT_RETAIL_TRAIN_INDICES[6:8])


def test_tau_bench_tasks_support_high_action_selection(tmp_path):
    tasks_train = tmp_path / "tasks_train.py"
    tasks_train.write_text(
        """
from tau_bench.types import Task, Action

TASKS_TRAIN = [
    Task(instruction="short", actions=[Action(name="a", kwargs={})], outputs=[]),
    Task(instruction="longer text", actions=[Action(name="a", kwargs={}), Action(name="b", kwargs={})], outputs=[]),
    Task(instruction="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", actions=[Action(name="a", kwargs={}), Action(name="b", kwargs={})], outputs=[]),
    Task(instruction="medium", actions=[Action(name="a", kwargs={}), Action(name="b", kwargs={}), Action(name="c", kwargs={})], outputs=[]),
]
"""
    )

    assert high_action_retail_train_indices(tasks_train_path=tasks_train, limit=3) == [3, 2, 1]

    out = tmp_path / "high.jsonl"
    report = materialize_tau_bench_tasks(
        out_jsonl=out,
        limit=2,
        offset=1,
        selection="high_action",
        tasks_train_path=tasks_train,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]

    assert report["selection"] == "high_action"
    assert report["task_indices"] == [2, 1]
    assert [row["grader"]["expected_answer"]["task_index"] for row in rows] == [2, 1]


def test_tau_bench_harness_registered():
    assert HARNESS_REGISTRY["tau_bench"] is TauBenchHarness


def test_tau_bench_turns_respect_workflow_budget():
    spec = task_spec(TauBenchTask(env_name="retail", task_split="train", task_index=351, max_turns=40))

    assert _max_turns_for_step(spec, "short") == 4
    assert _max_turns_for_step(spec, "medium") == 8
    assert _max_turns_for_step(spec, "long") == 20
    assert _max_turns_for_step(spec, "max") == 40


def _install_fake_tau_bench(monkeypatch):
    tau_mod = types.ModuleType("tau_bench")
    envs_mod = types.ModuleType("tau_bench.envs")
    types_mod = types.ModuleType("tau_bench.types")

    class Action:
        def __init__(self, name, kwargs):
            self.name = name
            self.kwargs = kwargs

    class Info:
        def model_dump(self, mode="json"):
            return {"fake": True}

    class Reset:
        observation = "Please look up the order and then respond."

    class StepResponse:
        def __init__(self, observation, reward, done):
            self.observation = observation
            self.reward = reward
            self.done = done
            self.info = Info()

    class FakeEnv:
        tools_info = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up the order.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        def __init__(self):
            self.user = None
            self.looked = False

        def reset(self, task_index=None):
            return Reset()

        def step(self, action):
            if action.name == "lookup_order":
                self.looked = True
                return StepResponse("order is eligible", 0.0, False)
            if action.name == "respond":
                return StepResponse("###STOP###", 1.0 if self.looked else 0.0, True)
            return StepResponse("bad action", 0.0, False)

    def get_env(*args, **kwargs):
        return FakeEnv()

    envs_mod.get_env = get_env
    types_mod.Action = Action
    monkeypatch.setitem(sys.modules, "tau_bench", tau_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.envs", envs_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.types", types_mod)


@pytest.mark.asyncio
async def test_tau_bench_harness_executes_tools_and_grades(monkeypatch, tmp_path):
    _install_fake_tau_bench(monkeypatch)
    spec = task_spec(TauBenchTask(env_name="retail", task_split="train", task_index=351, max_turns=4))
    calls = {"n": 0}

    def tool_fn(model, messages, tools, sampling):
        calls["n"] += 1
        if calls["n"] == 1:
            return ToolCompletion(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="lookup_order", arguments={})],
                model=model,
            )
        return ToolCompletion(content="Done.", tool_calls=[], model=model)

    pool = WorkerPool(
        [WorkerSpec(worker_id="tau_worker", model="fake/tau")],
        FakeProvider(tool_fn=tool_fn),
    )
    rec = await execute_workflow(
        spec,
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="solve the tau task", access=[])]),
        pool,
        Sampling(),
        "tau-ok",
        artifact_dir=tmp_path / "artifacts",
    )

    assert rec.grade and rec.grade.success is True
    assert rec.reward == 1.0
    assert rec.execution.steps[0].harness == "tau_bench"
    assert rec.execution.steps[0].messages_ref
