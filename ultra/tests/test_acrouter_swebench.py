import json

import pytest

from ultra.acrouter_swebench import SOURCE_NAME, materialize_swebench_ready_tasks, _redact_eval_log, _select_candidate
from ultra.registry import TaskRegistry
from ultra.schemas import SourceManifest


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_select_candidate_by_instance_id(tmp_path):
    ready = tmp_path / "ready.jsonl"
    _write_jsonl(
        ready,
        [
            {"task_id": "old112::a", "original_task_id": "a"},
            {"task_id": "old112::b", "original_task_id": "b"},
        ],
    )

    assert _select_candidate(ready)["original_task_id"] == "a"
    assert _select_candidate(ready, "b")["task_id"] == "old112::b"
    with pytest.raises(ValueError):
        _select_candidate(ready, "missing")


def test_redact_eval_log_removes_diff_but_keeps_test_markers():
    raw = """+ git -c core.fileMode=false diff abc123
diff --git a/foo.py b/foo.py
index 000..111 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-bad
+good
+ source /opt/miniconda3/bin/activate
: '>>>>> Start Test Output'
test output line
: '>>>>> End Test Output'
"""

    redacted = _redact_eval_log(raw)

    assert "bad" not in redacted
    assert "good" not in redacted
    assert "Start Test Output" in redacted
    assert "test output line" in redacted
    assert "[redacted" in redacted


def test_materialize_swebench_ready_tasks_as_pool_validation(tmp_path):
    ready = tmp_path / "ready.jsonl"
    _write_jsonl(
        ready,
        [
            {
                "candidate_id": "c1",
                "task_id": "old112::django__django-11292",
                "original_task_id": "django__django-11292",
                "success_count": 4,
                "model_count": 8,
                "disagreement_balance": 1.0,
                "swebench": {
                    "docker_image": "swebench/sweb.eval.x86_64.django_1776_django-11292:latest",
                    "difficulty": "1-4 hours",
                },
            }
        ],
    )
    report = materialize_swebench_ready_tasks(
        ready_jsonl=ready,
        out_jsonl=tmp_path / "tasks.jsonl",
        report_out=tmp_path / "report.json",
        swebench_rows={
            "django__django-11292": {
                "repo": "django/django",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug.",
            }
        },
    )

    rows = [json.loads(line) for line in (tmp_path / "tasks.jsonl").read_text().splitlines()]
    assert report["task_count"] == 1
    assert report["grpo_ready"] == 0
    assert rows[0]["source"]["name"] == SOURCE_NAME
    assert rows[0]["source"]["policy"] == "pool_only"
    assert rows[0]["splitting"]["split"] == "pool_validation"
    assert rows[0]["grader"]["type"] == "swebench_verified_hidden_tests"
    assert rows[0]["input"]["assets"][0]["opencode_instance"]["swebench_instance_id"] == "django__django-11292"
    assert rows[0]["environment"]["wall_time_seconds"] == 14400
    assert rows[0]["input"]["assets"][1]["acrouter_disagreement"]["difficulty"] == "1-4 hours"
    assert report["wall_time_by_task"][rows[0]["task_id"]] == 14400

    registry = TaskRegistry()
    registry.register_manifest(
        SourceManifest(
            source_name=SOURCE_NAME,
            source_type="public_benchmark_disagreement",
            version="ood176",
            allowed_uses=["pool_validation"],
            forbidden_uses=["grpo_train"],
        )
    )
    registry.add_many([])  # smoke empty iterator path
    from ultra.schemas import TaskSpec

    registry.add(TaskSpec.model_validate(rows[0]))
    assert len(registry) == 1
