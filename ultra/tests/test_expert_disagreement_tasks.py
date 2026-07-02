import json

from ultra.expert_disagreement_tasks import (
    CODE_TASKS,
    CODE_TASKS_V2,
    SOURCE_NAME,
    SOURCE_NAME_V2,
    code_task_spec,
    materialize_expert_disagreement_tasks,
    task_specs,
)
from ultra.grading.verifiers import get_grader
from ultra.registry import TaskRegistry
from ultra.schemas import SourceManifest


def test_expert_disagreement_tasks_materialize_and_ingest(tmp_path):
    out = tmp_path / "expert.jsonl"
    report = materialize_expert_disagreement_tasks(out_jsonl=out, report_out=tmp_path / "report.json")

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = task_specs()
    assert report["task_count"] == 14
    assert len(rows) == len(specs) == 14
    assert report["lane_counts"] == {
        "long_context_memory_planning": 2,
        "math_science_knowledge": 4,
        "tool_dialogue": 4,
        "unit_and_scientific_code": 4,
    }
    assert {row["source"]["name"] for row in rows} == {SOURCE_NAME}
    assert {row["splitting"]["split"] for row in rows} == {"grpo_train"}
    assert all("expert-designed" in row["metadata"]["tags"] for row in rows)
    assert all("disagreement-targeted" in row["metadata"]["tags"] for row in rows)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name=SOURCE_NAME,
            source_type="expert_designed_disagreement",
            version="v1",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(specs)


def test_expert_disagreement_v2_materialize_and_ingest(tmp_path):
    out = tmp_path / "expert_v2.jsonl"
    report = materialize_expert_disagreement_tasks(
        out_jsonl=out,
        report_out=tmp_path / "report_v2.json",
        version="v2",
    )

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    specs = task_specs(version="v2")
    assert report["task_count"] == 24
    assert len(rows) == len(specs) == 24
    assert report["source"] == SOURCE_NAME_V2
    assert report["lane_counts"] == {
        "long_context_memory_planning": 4,
        "math_science_knowledge": 6,
        "tool_dialogue": 6,
        "unit_and_scientific_code": 8,
    }
    assert {row["source"]["name"] for row in rows} == {SOURCE_NAME_V2}
    assert all("expert-designed" in row["metadata"]["tags"] for row in rows)
    assert all("disagreement-targeted" in row["metadata"]["tags"] for row in rows)

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name=SOURCE_NAME_V2,
            source_type="expert_designed_disagreement",
            version="v2",
            allowed_uses=["grpo_train"],
        )
    )
    registry.add_many(specs)
    assert len(registry) == len(specs)


def test_expert_code_task_verifier_catches_boundary_semantics():
    spec = code_task_spec(CODE_TASKS[1])
    grader = get_grader(spec.grader.type)
    correct = """
def merge_half_open(intervals):
    clean = sorted((a, b) for a, b in intervals if a < b)
    out = []
    for a, b in clean:
        if not out or a >= out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [tuple(x) for x in out]
"""
    closed_interval_bug = """
def merge_half_open(intervals):
    clean = sorted((a, b) for a, b in intervals if a < b)
    out = []
    for a, b in clean:
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [tuple(x) for x in out]
"""
    assert grader(correct, spec.grader.expected_answer) == 1.0
    assert grader(closed_interval_bug, spec.grader.expected_answer) == 0.0


def test_expert_v2_code_task_verifier_catches_ordering_semantics():
    task = next(t for t in CODE_TASKS_V2 if t.task_id == "feature-flags-last-write-delete")
    spec = code_task_spec(task, source_name=SOURCE_NAME_V2, source_version="v2")
    grader = get_grader(spec.grader.type)
    correct = """
def active_flags(events):
    first = []
    values = {}
    deleted = set()
    seen = set()
    for event in events:
        key = event["key"]
        if key not in seen:
            seen.add(key)
            first.append(key)
        if event.get("delete"):
            values.pop(key, None)
            deleted.add(key)
        else:
            values[key] = event.get("value")
            deleted.discard(key)
    return [(key, values[key]) for key in first if key in values and key not in deleted]
"""
    wrong_last_update_order = """
def active_flags(events):
    values = {}
    for event in events:
        key = event["key"]
        if event.get("delete"):
            values.pop(key, None)
        else:
            if key in values:
                del values[key]
            values[key] = event.get("value")
    return list(values.items())
"""
    assert grader(correct, spec.grader.expected_answer) == 1.0
    assert grader(wrong_last_update_order, spec.grader.expected_answer) == 0.0
