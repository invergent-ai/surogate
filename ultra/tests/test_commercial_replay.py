import json

from ultra.commercial_replay import build_commercial_replay


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _task(task_id):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "math",
        "source": {"name": "unit_test", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Solve 2+2."}]},
        "environment": {"harness": "direct_qa"},
        "grader": {"type": "exact_match", "expected_answer": "4"},
        "splitting": {"group_id": "unit_test", "split": "grpo_train"},
    }


def _rollout(task_id, reward=1.0, outcome="valid_correct_trainable"):
    return {
        "rollout_id": f"rollout_{task_id}",
        "task_id": task_id,
        "source_name": "unit_test",
        "capability": "math",
        "harness": "direct_qa",
        "conductor": {"checkpoint": None, "raw_output": None, "workflow_parse_valid": True},
        "workflow": {
            "steps": [
                {
                    "worker_id": 9,
                    "subtask": "Solve the problem.",
                    "access": [],
                    "budget": "medium",
                }
            ]
        },
        "execution": {
            "steps": [
                {
                    "worker_id": 9,
                    "harness": "direct_qa",
                    "budget": "medium",
                    "text": "4",
                    "termination": "completed",
                }
            ]
        },
        "grade": {"score": 1.0 if reward == 1.0 else 0.0, "success": reward == 1.0},
        "reward": reward,
        "outcome_class": outcome,
        "valid_for_training": True,
    }


def test_commercial_replay_exports_replay_and_success_sft(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    manifest_dir = repo_root / "director" / "manifests" / "fugu_clean_v1"
    run_dir = manifest_dir / "run_a"
    tasks_jsonl = manifest_dir / "tasks.jsonl"
    jobs_jsonl = run_dir / "jobs.jsonl"
    rollout_path = run_dir / "rollouts" / "job_00000.json"
    failed_rollout_path = run_dir / "rollouts" / "job_00001.json"

    _write_jsonl(tasks_jsonl, [_task("task_ok"), _task("task_bad")])
    _write_jsonl(
        jobs_jsonl,
        [
            {
                "job_id": "job_00000",
                "task_jsonl": str(tasks_jsonl),
                "source_task_id": "task_ok",
                "worker_names": ["direct_gpt_reasoner"],
            },
            {
                "job_id": "job_00001",
                "task_jsonl": str(tasks_jsonl),
                "source_task_id": "task_bad",
                "worker_names": ["direct_gpt_reasoner"],
            },
        ],
    )
    rollout_path.parent.mkdir(parents=True)
    rollout_path.write_text(json.dumps(_rollout("task_ok")), encoding="utf-8")
    failed_rollout_path.write_text(
        json.dumps(_rollout("task_bad", reward=0.5, outcome="valid_incorrect_trainable")),
        encoding="utf-8",
    )
    run_dir.joinpath("run_report.json").write_text(
        json.dumps(
            {
                "jobs_jsonl": str(jobs_jsonl),
                "rows": [
                    {
                        "job_id": "job_00000",
                        "task_id": "task_ok",
                        "source_task_id": "task_ok",
                        "lane": "math_science_knowledge",
                        "source": "unit_test",
                        "task_split": "grpo_train",
                        "arm": "solo__direct_gpt_reasoner",
                        "stage": "single_scaffold",
                        "worker_names": ["direct_gpt_reasoner"],
                        "rollout_out": str(rollout_path),
                        "outcome_class": "valid_correct_trainable",
                        "reward": 1.0,
                        "valid_for_training": True,
                    },
                    {
                        "job_id": "job_00001",
                        "task_id": "task_bad",
                        "source_task_id": "task_bad",
                        "lane": "math_science_knowledge",
                        "source": "unit_test",
                        "task_split": "grpo_train",
                        "arm": "solo__direct_gpt_reasoner",
                        "stage": "single_scaffold",
                        "worker_names": ["direct_gpt_reasoner"],
                        "rollout_out": str(failed_rollout_path),
                        "outcome_class": "valid_incorrect_trainable",
                        "reward": 0.5,
                        "valid_for_training": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    import ultra.commercial_replay as module

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    report = build_commercial_replay(manifest_dir, manifest_dir / "commercial_replay")

    assert report["counts"]["complete_commercial_replay_records"] == 2
    assert report["counts"]["successful_workflow_sft_records"] == 1

    replay_rows = [
        json.loads(line)
        for line in (manifest_dir / "commercial_replay" / "commercial_rollout_replay.jsonl").read_text().splitlines()
    ]
    sft_rows = [
        json.loads(line)
        for line in (manifest_dir / "commercial_replay" / "commercial_workflow_sft.jsonl").read_text().splitlines()
    ]

    assert replay_rows[0]["compact_workflow"]["steps"][0]["worker_id"] == 0
    assert sft_rows[0]["allowed_workers"][0]["name"] == "direct_gpt_reasoner"
    assert sft_rows[0]["workflow"]["steps"][0]["worker_id"] == 0
    assert json.loads(sft_rows[0]["messages"][-1]["content"]) == sft_rows[0]["workflow"]


def test_commercial_replay_skips_noncommercial_rows(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    manifest_dir = repo_root / "director" / "manifests" / "fugu_clean_v1"
    run_dir = manifest_dir / "run_b"
    rollout_path = run_dir / "rollouts" / "job_00000.json"
    rollout = _rollout("task_ok")
    rollout["workflow"]["steps"][0]["worker_id"] = 16
    rollout["execution"]["steps"][0]["worker_id"] = 16
    rollout_path.parent.mkdir(parents=True)
    rollout_path.write_text(json.dumps(rollout), encoding="utf-8")
    run_dir.joinpath("run_report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "job_id": "job_00000",
                        "task_id": "task_ok",
                        "worker_names": ["direct_glm_reasoner"],
                        "rollout_out": str(rollout_path),
                        "outcome_class": "valid_correct_trainable",
                        "reward": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    import ultra.commercial_replay as module

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    report = build_commercial_replay(manifest_dir, manifest_dir / "commercial_replay")

    assert report["counts"]["complete_commercial_replay_records"] == 0
    assert report["counts"]["non_commercial_rows"] == 1
