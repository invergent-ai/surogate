import json

import pyarrow as pa
import pyarrow.parquet as pq

from ultra.agenttrove_disagreement import scan_agenttrove_disagreement, suggested_tasktrove_source


def _write_parquet(path, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _taskspec(source, task_id):
    return {
        "schema_version": "2.0",
        "task_id": f"{source}__{task_id}",
        "source": {"name": source, "policy": "train_allowed"},
        "splitting": {"split": "grpo_train"},
    }


def test_agenttrove_disagreement_report_ranks_mixed_outcome_tasks(tmp_path):
    parquet = _write_parquet(
        tmp_path / "agenttrove.parquet",
        [
            {
                "original_source": "Inferred Bugs",
                "original_teacher": "Teacher A",
                "model": "model-a",
                "task": "inferredbugs-0001",
                "reward": 1.0,
            },
            {
                "original_source": "Inferred Bugs",
                "original_teacher": "Teacher B",
                "model": "model-b",
                "task": "inferredbugs-0001",
                "reward": 0.0,
            },
            {
                "original_source": "Inferred Bugs",
                "original_teacher": "Teacher A",
                "model": "model-a",
                "task": "inferredbugs-0002",
                "reward": 1.0,
            },
            {
                "original_source": "Inferred Bugs",
                "original_teacher": "Teacher B",
                "model": "model-b",
                "task": "inferredbugs-0002",
                "reward": 1.0,
            },
            {
                "original_source": "nl2bash",
                "original_teacher": "Teacher C",
                "model": "model-c",
                "task": "nl2bash-0001",
                "reward": None,
            },
        ],
    )
    manifest_dir = tmp_path / "manifest"
    _write_jsonl(
        manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl",
        [_taskspec("tasktrove_inferredbugs", "inferredbugs-0001")],
    )

    report = scan_agenttrove_disagreement(
        parquet_paths=[parquet],
        candidates_out=tmp_path / "candidates.jsonl",
        report_out=tmp_path / "report.json",
        manifest_dir=manifest_dir,
    )

    candidates = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert report["rows_scanned"] == 5
    assert report["rows_with_outcome"] == 4
    assert report["disagreement_candidates"] == 1
    assert candidates[0]["task_id"] == "inferredbugs-0001"
    assert candidates[0]["suggested_tasktrove_source"] == "tasktrove_inferredbugs"
    assert candidates[0]["local_tasktrove_exact_match"] is True
    assert candidates[0]["success_count"] == 1
    assert candidates[0]["failure_count"] == 1


def test_agenttrove_disagreement_report_supports_result_strings_and_source_filter(tmp_path):
    parquet = _write_parquet(
        tmp_path / "agenttrove.parquet",
        [
            {
                "original_source": "nl2bash",
                "original_teacher": "Teacher A",
                "task_id": "nl2bash-1",
                "result": "passed",
            },
            {
                "original_source": "nl2bash",
                "original_teacher": "Teacher B",
                "task_id": "nl2bash-1",
                "result": "failed",
            },
            {
                "original_source": "freelancer",
                "original_teacher": "Teacher C",
                "task_id": "freelancer-1",
                "result": "failed",
            },
        ],
    )

    report = scan_agenttrove_disagreement(
        parquet_paths=[parquet],
        candidates_out=tmp_path / "candidates.jsonl",
        source_filter={"nl2bash"},
    )

    candidates = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert report["rows_used"] == 2
    assert report["disagreement_candidates"] == 1
    assert candidates[0]["suggested_tasktrove_source"] == "tasktrove_nl2bash"


def test_agenttrove_disagreement_report_can_use_self_reported_completion_prior(tmp_path):
    parquet = _write_parquet(
        tmp_path / "agenttrove.parquet",
        [
            {
                "original_source": "r2egym",
                "original_teacher": "Teacher A",
                "task": "r2egym-1",
                "result": None,
                "conversations": [
                    {"role": "assistant", "content": '{"task_complete": true}'},
                ],
            },
            {
                "original_source": "r2egym",
                "original_teacher": "Teacher B",
                "task": "r2egym-1",
                "result": None,
                "conversations": [
                    {"role": "assistant", "content": '{"task_complete": false}'},
                ],
            },
        ],
    )

    report = scan_agenttrove_disagreement(
        parquet_paths=[parquet],
        candidates_out=tmp_path / "candidates.jsonl",
        allow_self_reported_completion=True,
    )

    candidates = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert report["outcome_source_counts"] == {"self_reported_task_complete": 2}
    assert report["disagreement_candidates"] == 1
    assert candidates[0]["outcome_sources"] == {"self_reported_task_complete": 2}
    assert candidates[0]["use_policy"] == "prioritize_for_local_tasktrove_prefilter_not_grpo_label"


def test_suggested_tasktrove_source_normalizes_known_aliases():
    assert suggested_tasktrove_source("Inferred Bugs") == "tasktrove_inferredbugs"
    assert suggested_tasktrove_source("SWE-Smith") == "tasktrove_swesmith"
