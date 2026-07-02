import io
import json
import tarfile

import pytest

from ultra.tasktrove_prefilter import (
    build_agenttrove_exact_prefilter_batch,
    build_tasktrove_prefilter_batch,
    build_tasktrove_reservoir_report,
    select_agenttrove_exact_matches,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _task(tmp_path, source, task_id, content="Solve it."):
    task_dir = tmp_path / "harbor" / source / task_id
    task_dir.mkdir(parents=True)
    return {
        "schema_version": "2.0",
        "task_id": f"{source}__{task_id}",
        "capability": "terminal_agentic",
        "source": {"name": source, "version": "v1", "policy": "train_allowed"},
        "input": {
            "messages": [{"role": "user", "content": content}],
            "assets": [{"harbor_task": {"task_dir": str(task_dir), "agent": "terminus-2"}}],
        },
        "environment": {"harness": "terminal_sandbox", "wall_time_seconds": 900},
        "grader": {"type": "harbor_verifier", "success_threshold": 1.0},
        "splitting": {
            "group_id": source,
            "split": "grpo_train",
            "contamination_group": f"{source}/{task_id}",
        },
        "metadata": {"domain": "terminal", "tags": ["tasktrove", "verifier_backed"]},
    }


def _exact_match(folder, task_path, *, teacher_count=3, model_count=2, success_rate=0.5, attempts=6):
    success_count = round(attempts * success_rate)
    failure_count = attempts - success_count
    return {
        "candidate_id": f"agenttrove::{folder}::{task_path}",
        "tasktrove_folder": folder,
        "tasktrove_path": task_path,
        "task_id": task_path,
        "attempts": attempts,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / attempts,
        "teacher_count": teacher_count,
        "teachers": {f"teacher-{i}": 1 for i in range(teacher_count)},
        "models": {f"model-{i}": 1 for i in range(model_count)},
    }


def _harbor_task_tar_bytes(task_id):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        files = {
            "instruction.md": "Solve it.",
            "task.toml": f"""
[metadata]
task_id = "{task_id}"
category = "test"

[environment]
docker_image = "python:3.11"

[verifier]
timeout_sec = 10
""",
            "tests/test_placeholder.py": "def test_placeholder():\n    assert True\n",
        }
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return payload.getvalue()


def test_build_tasktrove_prefilter_batch_excludes_prior_prefilter_tasks(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    inferred = [_task(tmp_path, "tasktrove_inferredbugs", f"inferredbugs-{i:04d}") for i in range(4)]
    pymethods = [_task(tmp_path, "tasktrove_pymethods2test", f"pymethods2test-{i:04d}") for i in range(4)]
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl", inferred)
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl", pymethods)
    _write_jsonl(
        manifest_dir / "tasktrove_prefilter_old" / "taskspecs.jsonl",
        [inferred[0], pymethods[0]],
    )

    report = build_tasktrove_prefilter_batch(
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "tasktrove_prefilter_new",
        inferredbugs_count=2,
        pymethods_count=2,
        seed=7,
    )

    assert report["selected_tasks"] == 4
    assert report["selected_source_counts"] == {
        "tasktrove_inferredbugs": 2,
        "tasktrove_pymethods2test": 2,
    }
    assert "tasktrove_inferredbugs__inferredbugs-0000" not in report["sources"]["tasktrove_inferredbugs"][
        "selected_task_ids"
    ]
    manifest = json.loads((manifest_dir / "tasktrove_prefilter_new" / "scaffold_tournament_manifest.json").read_text())
    assert manifest["selected_task_counts"] == {
        "repo_open_repo_terminal": 2,
        "unit_and_scientific_code": 2,
    }
    assert manifest["selected_arm_domain_counts"] == {"terminal_sandbox": 4}
    assert report["job_count"] == 36
    assert report["ready_jobs"] == 36
    assert (manifest_dir / "tasktrove_prefilter_new" / "scaffold_tournament_jobs.jsonl").exists()


def test_ranked_tasktrove_prefilter_prefers_complex_diverse_rows(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    easy = "Solve it."
    hard = "Implement the function with constraints, edge cases, input/output handling, tests, and file updates. " * 20
    harder = hard + ("Additional hidden edge cases and regression tests. " * 10)
    inferred = [
        _task(tmp_path, "tasktrove_inferredbugs", "inferredbugs-0001", easy),
        _task(tmp_path, "tasktrove_inferredbugs", "inferredbugs-0002", easy),
        _task(tmp_path, "tasktrove_inferredbugs", "inferredbugs-0010", harder),
        _task(tmp_path, "tasktrove_inferredbugs", "inferredbugs-0011", hard),
        _task(tmp_path, "tasktrove_inferredbugs", "inferredbugs-0020", hard),
    ]
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl", inferred)
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl", [])

    report = build_tasktrove_prefilter_batch(
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "tasktrove_prefilter_ranked",
        inferredbugs_count=2,
        pymethods_count=0,
        seed=11,
    )

    selected = report["sources"]["tasktrove_inferredbugs"]["selected_task_ids"]
    assert selected == [
        "tasktrove_inferredbugs__inferredbugs-0010",
        "tasktrove_inferredbugs__inferredbugs-0020",
    ]
    assert report["selection_policy"]["mode"] == "ranked"
    diagnostics = report["sources"]["tasktrove_inferredbugs"]["selected_task_diagnostics"]
    assert diagnostics[0]["features"]["prompt_chars"] > len(easy)


def test_sequential_tasktrove_prefilter_mode_remains_available(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    inferred = [_task(tmp_path, "tasktrove_inferredbugs", f"inferredbugs-{i:04d}") for i in range(3)]
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl", inferred)
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl", [])

    report = build_tasktrove_prefilter_batch(
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "tasktrove_prefilter_sequential",
        inferredbugs_count=2,
        pymethods_count=0,
        seed=11,
        selection="sequential",
    )

    assert report["sources"]["tasktrove_inferredbugs"]["selected_task_ids"] == [
        "tasktrove_inferredbugs__inferredbugs-0000",
        "tasktrove_inferredbugs__inferredbugs-0001",
    ]
    assert report["selection_policy"]["mode"] == "sequential"


def test_tasktrove_prefilter_accepts_dynamic_sources(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    source = "tasktrove_r2egym"
    tasks = [_task(tmp_path, source, f"r2egym-{i:04d}") for i in range(5)]
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "diversity" / "r2egym" / "taskspecs.jsonl", tasks)

    report = build_tasktrove_prefilter_batch(
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "tasktrove_prefilter_dynamic",
        source_counts={source: 3},
        seed=13,
    )

    assert report["selected_tasks"] == 3
    assert report["selected_source_counts"] == {source: 3}
    assert report["sources"][source]["available_after_seen"] == 5
    assert report["job_count"] == 27


def test_tasktrove_reservoir_report_counts_available_rows(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    source = "tasktrove_nl2bash"
    tasks = [_task(tmp_path, source, f"nl2bash-{i:04d}") for i in range(4)]
    _write_jsonl(manifest_dir / "tasktrove_harbor" / "diversity" / "nl2bash" / "taskspecs.jsonl", tasks)
    _write_jsonl(manifest_dir / "tasktrove_prefilter_old" / "taskspecs.jsonl", [tasks[0]])

    report = build_tasktrove_reservoir_report(
        manifest_dir=manifest_dir,
        report_out=tmp_path / "reservoir.json",
    )

    assert report["source_count"] == 1
    assert report["total_train_allowed_tasks"] == 4
    assert report["total_seen_tasks"] == 1
    assert report["total_available_tasks"] == 3
    assert report["sources"][source]["available"] == 3


def test_select_agenttrove_exact_matches_prefers_diverse_unseen_rows(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    exact_matches = manifest_dir / "agenttrove_disagreement" / "exact.jsonl"
    folder = "DCAgent__r2egym-patched-full-oracle"
    rows = [
        _exact_match(folder, "r2egym-v1-00001", teacher_count=5, model_count=5, attempts=8),
        _exact_match(folder, "r2egym-v1-00002", teacher_count=1, model_count=1, attempts=8),
        _exact_match(folder, "r2egym-v1-00003", teacher_count=4, model_count=3, success_rate=1.0),
        _exact_match(folder, "r2egym-v1-00004", teacher_count=4, model_count=3, attempts=6),
    ]
    _write_jsonl(exact_matches, rows)
    _write_jsonl(
        manifest_dir / "agenttrove_disagreement" / "exact_match_selection_001" / f"{folder}.jsonl",
        [rows[0]],
    )

    report = select_agenttrove_exact_matches(
        exact_matches_jsonl=exact_matches,
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "agenttrove_disagreement" / "exact_match_selection_002",
        source_counts={"tasktrove_r2egym": 2},
        seed=1,
        min_teacher_count=3,
        min_model_count=2,
    )

    selected = report["sources"]["tasktrove_r2egym"]["selected_tasktrove_paths"]
    assert selected == ["r2egym-v1-00004"]
    assert report["sources"]["tasktrove_r2egym"]["deficit"] == 1
    assert report["selected_tasks"] == 1


def test_build_agenttrove_exact_prefilter_batch_materializes_local_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    manifest_dir = tmp_path / "fugu_clean_v1"
    tasktrove_root = tmp_path / "tasktrove"
    folder = "DCAgent__r2egym-patched-full-oracle"
    parquet_dir = tasktrove_root / folder
    parquet_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "path": ["r2egym-v1-00001", "r2egym-v1-00002"],
                "task_binary": [_harbor_task_tar_bytes("r2egym-v1-00001"), _harbor_task_tar_bytes("r2egym-v1-00002")],
            }
        ),
        parquet_dir / "tasks.parquet",
    )

    exact_matches = manifest_dir / "agenttrove_disagreement" / "exact.jsonl"
    _write_jsonl(
        exact_matches,
        [
            _exact_match(folder, "r2egym-v1-00001", teacher_count=5, model_count=5),
            _exact_match(folder, "r2egym-v1-00002", teacher_count=4, model_count=3),
        ],
    )

    report = build_agenttrove_exact_prefilter_batch(
        exact_matches_jsonl=exact_matches,
        tasktrove_root=tasktrove_root,
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "tasktrove_prefilter_agenttrove_exact_002",
        source_counts={"tasktrove_r2egym": 1},
        seed=2,
        min_teacher_count=3,
        min_model_count=2,
    )

    assert report["selected_tasks"] == 1
    assert report["materialized_tasks"] == 1
    assert report["materialized_source_counts"] == {"tasktrove_r2egym": 1}
    assert report["job_count"] == 9
    assert report["ready_jobs"] == 9
    assert (manifest_dir / "tasktrove_prefilter_agenttrove_exact_002" / "scaffold_tournament_jobs.jsonl").exists()


def test_agenttrove_exact_code_contests_fill_unit_lane_with_terminal_arms(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    manifest_dir = tmp_path / "fugu_clean_v1"
    tasktrove_root = tmp_path / "tasktrove"
    folder = "DCAgent__code-contests-noblock"
    parquet_dir = tasktrove_root / folder
    parquet_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "path": ["code_contests-0001"],
                "task_binary": [_harbor_task_tar_bytes("code_contests-0001")],
            }
        ),
        parquet_dir / "tasks.parquet",
    )

    exact_matches = manifest_dir / "agenttrove_disagreement" / "exact.jsonl"
    _write_jsonl(
        exact_matches,
        [_exact_match(folder, "code_contests-0001", teacher_count=5, model_count=4)],
    )

    report = build_agenttrove_exact_prefilter_batch(
        exact_matches_jsonl=exact_matches,
        tasktrove_root=tasktrove_root,
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "tasktrove_prefilter_agenttrove_code",
        source_counts={"tasktrove_code_contests": 1},
        seed=3,
        min_teacher_count=3,
        min_model_count=2,
    )

    manifest = json.loads(
        (manifest_dir / "tasktrove_prefilter_agenttrove_code" / "scaffold_tournament_manifest.json").read_text()
    )
    assert report["selected_source_counts"] == {"tasktrove_code_contests": 1}
    assert manifest["selected_task_counts"] == {"unit_and_scientific_code": 1}
    assert manifest["selected_arm_domain_counts"] == {"terminal_sandbox": 1}
    assert report["job_count"] == 9
