import csv
import json

from ultra.acrouter_candidates import build_ood176_reconstruction_queue, extract_ood176_disagreement_candidates


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_acrouter_ood176_candidates_extract_partial_solves(tmp_path):
    bench = tmp_path / "coderouterbench"
    bench.mkdir()
    _write_jsonl(
        bench / "ood176_tasks.jsonl",
        [
            {
                "task_id": "t_partial",
                "source_split": "new64",
                "bench": "featurebench",
                "original_task_id": "repo.issue",
                "dimension": "bug_fixing",
                "language": "python",
                "difficulty": "hard",
                "prompt": "Fix the bug.",
            },
            {"task_id": "t_all_fail", "bench": "longcli", "dimension": "code_generation"},
            {"task_id": "t_all_solve", "bench": "old112", "dimension": "bug_fixing"},
        ],
    )
    with (bench / "ood176_results_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "source_split",
                "bench",
                "original_task_id",
                "dimension",
                "model",
                "source_model",
                "resolved",
                "apply_ok",
                "graded",
            ],
        )
        writer.writeheader()
        for task_id, scores in {
            "t_partial": [1, 0],
            "t_all_fail": [0, 0],
            "t_all_solve": [1, 1],
        }.items():
            for index, score in enumerate(scores):
                writer.writerow(
                    {
                        "task_id": task_id,
                        "model": f"m{index}",
                        "resolved": str(score),
                        "apply_ok": "1",
                        "graded": "1",
                    }
                )

    report = extract_ood176_disagreement_candidates(
        bench,
        tmp_path / "candidates.jsonl",
        tmp_path / "report.json",
        index_out=tmp_path / "index.jsonl",
        md_out=tmp_path / "report.md",
    )

    rows = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert report["partial_solve_disagreement_candidates"] == 1
    assert report["all_fail_tasks"] == 1
    assert report["all_solve_tasks"] == 1
    assert rows[0]["task_id"] == "t_partial"
    assert rows[0]["grpo_ready"] is False
    assert rows[0]["success_count"] == 1
    assert rows[0]["model_count"] == 2
    assert "executable_grader" in rows[0]["requires_reconstruction"]


def test_acrouter_reconstruction_queue_marks_ready_swebench_inputs(tmp_path):
    candidates = [
        {
            "candidate_id": "c1",
            "task_id": "old112::django__django-10973",
            "bench": "old112",
            "original_task_id": "django__django-10973",
            "source_dataset": "SWE-bench Verified",
            "dimension": "bug_fixing",
            "prompt": "Fix issue",
            "success_count": 2,
            "model_count": 8,
        },
        {
            "candidate_id": "c2",
            "task_id": "new64::featurebench::repo.issue",
            "bench": "featurebench",
            "original_task_id": "repo.issue",
            "dimension": "bug_fixing",
            "prompt": "Feature task",
            "success_count": 4,
            "model_count": 8,
        },
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    _write_jsonl(candidate_path, candidates)

    image = "swebench/sweb.eval.x86_64.django_1776_django-10973:latest"
    report = build_ood176_reconstruction_queue(
        candidate_path,
        tmp_path / "queue.jsonl",
        tmp_path / "report.json",
        md_out=tmp_path / "report.md",
        ready_swebench_out=tmp_path / "ready.jsonl",
        ready_swebench_report_out=tmp_path / "ready_report.json",
        swebench_index={
            "django__django-10973": {
                "repo": "django/django",
                "base_commit": "abc123",
                "difficulty": "hard",
            }
        },
        docker_images={image},
    )

    rows = [json.loads(line) for line in (tmp_path / "queue.jsonl").read_text().splitlines()]
    assert report["ready_for_swebench_adapter_validation"] == 1
    assert rows[0]["task_id"] == "old112::django__django-10973"
    assert rows[0]["reconstruction_status"] == "ready_for_swebench_adapter_validation"
    assert rows[0]["permitted_use"] == "pool_validation_only"
    assert rows[0]["grpo_ready"] is False
    assert rows[0]["swebench"]["dataset_row_available"] is True
    assert rows[0]["swebench"]["docker_image_available"] is True
    assert rows[1]["reconstruction_status"] == "needs_featurebench_harness"
    ready = [json.loads(line) for line in (tmp_path / "ready.jsonl").read_text().splitlines()]
    ready_report = json.loads((tmp_path / "ready_report.json").read_text())
    assert len(ready) == 1
    assert ready_report["instance_ids"] == ["django__django-10973"]
