import json

from ultra.label_prior_selection import build_label_prior_shard


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _task(task_id, domain="code", source_family="taco"):
    harness = "code_exec" if domain == "code" else "direct_qa"
    capability = "unit_code" if domain == "code" else "math"
    return {
        "schema_version": "2.0",
        "task_id": f"existing_bank__{task_id}",
        "capability": capability,
        "source": {"name": "existing_bank", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Solve it."}]},
        "environment": {"harness": harness},
        "grader": {"type": "code_exec_stdio" if domain == "code" else "math_equal", "expected_answer": "42"},
        "splitting": {
            "group_id": source_family,
            "split": "grpo_train",
            "contamination_group": f"{source_family}::{task_id}",
        },
        "metadata": {"domain": domain, "tags": ["existing_bank"]},
    }


def _label(task_id, r_bar, domain="code", source="taco"):
    return {
        "task_id": task_id,
        "domain": domain,
        "source": source,
        "prompt": "Solve it.",
        "worker_ids": ["deepseek", "kimi", "glm", "mimo", "minimax", "deepseek_flash"],
        "r_bar": r_bar,
        "p": [0.1] * 6,
        "grader": "code_exec_stdio" if domain == "code" else "math_equal",
    }


def test_label_prior_shard_selects_fresh_unit_code_and_writes_manifest(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    _write_jsonl(
        manifest_dir / "data_mix" / "existing_bank_taskspecs.jsonl",
        [
            _task("seen"),
            _task("attempted"),
            _task("selected-a", source_family="taco"),
            _task("selected-b", source_family="code_contests"),
            _task("easy"),
            _task("math-1", domain="math", source_family="numina_math"),
        ],
    )
    _write_jsonl(
        manifest_dir / "labels_n4_tau0.1.jsonl",
        [
            _label("seen", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            _label("attempted", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            _label("selected-a", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], source="taco"),
            _label("selected-b", [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], source="code_contests"),
            _label("easy", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            _label("math-1", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], domain="math", source="numina_math"),
        ],
    )
    _write_jsonl(
        manifest_dir / "pool_matrix_frontier.jsonl",
        [
            {
                "task_id": "selected-a",
                "domain": "code",
                "worker_ids": ["opus", "gemini", "gpt", "glm"],
                "r_bar": [1.0, 0.5, 0.0, 0.25],
            }
        ],
    )
    _write_jsonl(
        manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl",
        [{"source": "existing_bank", "source_task_id": "existing_bank__seen"}],
    )
    _write_jsonl(manifest_dir / "label_prior_old" / "taskspecs.jsonl", [_task("attempted")])

    report = build_label_prior_shard(
        manifest_dir=manifest_dir,
        out_dir=manifest_dir / "label_prior_new",
        lane="unit_and_scientific_code",
        count=2,
        seed=5,
    )

    assert report["selected"] == 2
    assert report["ready_jobs"] == report["job_count"]
    selected_ids = [
        row["task_id"] for row in _read_jsonl(manifest_dir / "label_prior_new" / "selected_candidates.jsonl")
    ]
    assert selected_ids == ["selected-a", "selected-b"]
    task_rows = _read_jsonl(manifest_dir / "label_prior_new" / "taskspecs.jsonl")
    assert {row["task_id"] for row in task_rows} == {
        "existing_bank__selected-a",
        "existing_bank__selected-b",
    }
    selected_a = next(row for row in task_rows if row["task_id"] == "existing_bank__selected-a")
    assert selected_a["metadata"]["label_prior"]["source_file"] == "labels_n4_tau0.1.jsonl"
    assert selected_a["metadata"]["frontier_matrix_prior"]["source_file"] == "pool_matrix_frontier.jsonl"
    manifest = json.loads((manifest_dir / "label_prior_new" / "scaffold_tournament_manifest.json").read_text())
    assert manifest["selected_task_counts"] == {"unit_and_scientific_code": 2}
    assert all(job["lane"] == "unit_and_scientific_code" for job in manifest["jobs"])


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
