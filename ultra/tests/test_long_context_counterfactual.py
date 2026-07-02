import json

from ultra.grading.verifiers import get_grader
from ultra.long_context_counterfactual_tasks import (
    TASKS,
    build_counterfactual_tasks,
    materialize_long_context_counterfactual_tasks,
    task_spec,
)


def test_counterfactual_tasks_validate_and_grade(tmp_path):
    assert len(TASKS) == 24
    seen = set()
    grader = get_grader("contains_all_absent")

    for task in TASKS:
        assert task.task_id not in seen
        seen.add(task.task_id)
        spec = task_spec(task)
        expected = spec.grader.expected_answer
        gold = task.must_contain[0]

        assert spec.environment.harness == "long_context"
        assert spec.splitting.split == "grpo_train"
        assert len(spec.input.context_documents) >= 6
        assert grader(gold, expected) == 1.0

        forbidden = task.must_not_contain[0]
        assert grader(f"{gold} {forbidden}", expected) == 0.0

    out = tmp_path / "taskspecs.jsonl"
    report = materialize_long_context_counterfactual_tasks(out_jsonl=out, report_out=tmp_path / "report.json")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert report["task_count"] == 24
    assert len(rows) == 24


def test_counterfactual_offset_groups_are_unique():
    first = {task.task_id for task in build_counterfactual_tasks(start=0, groups=2)}
    second = {task.task_id for task in build_counterfactual_tasks(start=2, groups=2)}

    assert len(first) == 8
    assert len(second) == 8
    assert first.isdisjoint(second)
