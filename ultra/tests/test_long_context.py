import json

import pytest

from ultra.config import WorkerSpec
from ultra.executor import execute_workflow
from ultra.grading.verifiers import contains_all_absent
from ultra.harness import HARNESS_REGISTRY
from ultra.harness.long_context import LongContextHarness
from ultra.long_context_adversarial_tasks import (
    TASKS as ADVERSARIAL_TASKS,
    materialize_long_context_adversarial_tasks,
    task_spec as adversarial_task_spec,
)
from ultra.long_context_stress_tasks import (
    TASKS as STRESS_TASKS,
    materialize_long_context_stress_tasks,
    task_spec as stress_task_spec,
)
from ultra.long_context_tasks import TASKS, materialize_long_context_tasks, task_spec
from ultra.registry import TaskRegistry
from ultra.schemas import SourceManifest, Workflow, WorkflowStep
from ultra.workers import FakeProvider, Sampling, WorkerPool


def test_long_context_tasks_materialize_and_ingest(tmp_path):
    out = tmp_path / "tasks.jsonl"
    report = materialize_long_context_tasks(out_jsonl=out, report_out=tmp_path / "report.json")

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = [task_spec(task) for task in TASKS]
    hard_rows = [row for row in rows if row["task_id"].endswith(tuple(f"hard-{suffix}" for suffix in [
        "security-exception-summary",
        "release-rollback-summary",
        "experiment-routing-summary",
        "incident-action-summary",
        "procurement-renewal-summary",
        "customer-pilot-summary",
    ]))]
    assert report["task_count"] == len(TASKS)
    assert len(rows) == len(TASKS)
    assert len(hard_rows) == 6
    assert all("hard" in row["metadata"]["tags"] for row in hard_rows)
    assert all("multi_hop" in row["metadata"]["tags"] for row in hard_rows)
    assert all(len(row["input"]["context_documents"]) >= 5 for row in hard_rows)
    assert all(" / " in row["grader"]["expected_answer"] for row in hard_rows)
    assert {row["source"]["name"] for row in rows} == {"longctx_generated"}
    assert {row["environment"]["harness"] for row in rows} == {"long_context"}
    assert all(row["input"]["context_documents"] for row in rows)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name="longctx_generated",
            source_type="generated_long_context",
            version="v1",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(TASKS)


def test_adversarial_long_context_tasks_materialize_and_ingest(tmp_path):
    out = tmp_path / "adversarial.jsonl"
    report = materialize_long_context_adversarial_tasks(out_jsonl=out, report_out=tmp_path / "report.json")

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = [adversarial_task_spec(task) for task in ADVERSARIAL_TASKS]
    assert report["task_count"] == len(ADVERSARIAL_TASKS)
    assert len(rows) == len(ADVERSARIAL_TASKS)
    assert {row["source"]["name"] for row in rows} == {"longctx_adversarial"}
    assert {row["environment"]["harness"] for row in rows} == {"long_context"}
    assert {row["grader"]["type"] for row in rows} == {"contains_all_absent"}
    assert all(row["grader"]["expected_answer"]["must_contain"] for row in rows)
    assert all(row["grader"]["expected_answer"]["must_not_contain"] for row in rows)
    assert all("adversarial" in row["metadata"]["tags"] for row in rows)
    assert all(len(row["input"]["context_documents"]) >= 9 for row in rows)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name="longctx_adversarial",
            source_type="generated_long_context",
            version="v1",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(ADVERSARIAL_TASKS)


def test_stress_long_context_tasks_materialize_and_ingest(tmp_path):
    out = tmp_path / "stress.jsonl"
    report = materialize_long_context_stress_tasks(out_jsonl=out, report_out=tmp_path / "report.json")

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = [stress_task_spec(task) for task in STRESS_TASKS]
    assert report["task_count"] == len(STRESS_TASKS)
    assert len(rows) == len(STRESS_TASKS)
    assert {row["source"]["name"] for row in rows} == {"longctx_stress"}
    assert {row["environment"]["harness"] for row in rows} == {"long_context"}
    assert {row["grader"]["type"] for row in rows} == {"contains_all_absent"}
    assert all("stress" in row["metadata"]["tags"] for row in rows)
    assert all("arithmetic" in row["metadata"]["tags"] for row in rows)
    assert all(row["grader"]["expected_answer"]["must_contain"] for row in rows)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name="longctx_stress",
            source_type="generated_long_context",
            version="v1",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(STRESS_TASKS)


def test_long_context_harness_registered():
    assert HARNESS_REGISTRY["long_context"] is LongContextHarness


def test_contains_all_absent_judges_final_matching_line():
    solution = {"must_contain": ["Kestrel", "742", "AP-77"], "must_not_contain": ["Northwind"]}
    verbose = "Northwind was ignored during reconciliation.\n\nKestrel / 742 / AP-77"
    stale_final = "Kestrel / 742 / AP-77 / Northwind"
    assert contains_all_absent(verbose, solution) == 1.0
    assert contains_all_absent(stale_final, solution) == 0.0


@pytest.mark.asyncio
async def test_long_context_harness_supplies_documents_and_grades_success():
    spec = task_spec(TASKS[0])

    def answer_fn(model, messages, sampling):
        joined = "\n".join(str(m["content"]) for m in messages)
        assert "amber-lattice" in joined
        return "The assigned codename was amber-lattice."

    pool = WorkerPool(
        [WorkerSpec(worker_id="longctx_worker", model="fake/longctx")],
        FakeProvider(answer_fn),
    )
    rec = await execute_workflow(
        spec,
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="answer from documents", access=[])]),
        pool,
        Sampling(),
        "longctx-ok",
    )

    assert rec.grade and rec.grade.success is True
    assert rec.reward == 1.0
    assert rec.execution.steps[0].harness == "long_context"


@pytest.mark.asyncio
async def test_adversarial_long_context_grader_rejects_stale_values():
    spec = adversarial_task_spec(ADVERSARIAL_TASKS[0])

    def answer_fn(model, messages, sampling):
        joined = "\n".join(str(m["content"]) for m in messages)
        assert "vault-7" in joined
        return "Mira Chen, Theo Martin / Nadia Flores / rot-7319"

    pool = WorkerPool(
        [WorkerSpec(worker_id="longctx_worker", model="fake/longctx")],
        FakeProvider(answer_fn),
    )
    rec = await execute_workflow(
        spec,
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="answer from documents", access=[])]),
        pool,
        Sampling(),
        "longctx-adversarial-ok",
    )
    assert rec.grade and rec.grade.success is True
    assert rec.reward == 1.0

    def stale_answer_fn(model, messages, sampling):
        return "Mira Chen, Theo Martin / Nadia Flores / rot-7319; old value rot-1190 was superseded."

    stale_pool = WorkerPool(
        [WorkerSpec(worker_id="longctx_worker", model="fake/longctx")],
        FakeProvider(stale_answer_fn),
    )
    stale = await execute_workflow(
        spec,
        Workflow(steps=[WorkflowStep(worker_id=0, subtask="answer from documents", access=[])]),
        stale_pool,
        Sampling(),
        "longctx-adversarial-stale",
    )
    assert stale.grade and stale.grade.success is False
    assert stale.reward == 0.5
