import json

from ultra.grpo_pilot_gap_plan import build_grpo_pilot_gap_plan


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")


def _rollout(task_id, lane, source, arm, stage, reward):
    return {
        "job_id": f"{task_id}-{arm}",
        "tournament_task_id": f"{lane}::{source}::{task_id}",
        "lane": lane,
        "source": source,
        "source_task_id": task_id,
        "task_jsonl": "/tmp/tasks.jsonl",
        "arm": arm,
        "stage": stage,
        "worker_names": ["w"],
        "reward": reward,
        "valid_for_training": True,
    }


def test_build_grpo_pilot_gap_plan_reports_deficits_and_blockers(tmp_path):
    manifest_dir = tmp_path / "manifest"
    seed_jsonl = manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl"
    _write_jsonl(
        seed_jsonl,
        [
            {"lane": "tool_dialogue", "source": "tau_bench_retail_train"},
            {"lane": "repo_open_repo_terminal", "source": "tasktrove_inferredbugs"},
        ],
    )
    _write_json(
        manifest_dir / "taskcraft_source_probe" / "report.json",
        {
            "status": "candidate_source_not_grpo_ready",
            "candidate_count": 12,
            "candidate_count_before_limit": 120,
            "raw_dataset_grpo_ready": False,
            "readiness_blockers": ["source_documents_not_frozen"],
        },
    )
    _write_json(
        manifest_dir / "taskcraft_source_probe" / "readiness_audit.json",
        {
            "status": "audited_not_grpo_ready",
            "freeze_priority_count": 4,
            "linkage_counts": {"any_atomic_match": 8},
            "readiness_blocker_counts": {"source_documents_not_frozen": 12},
            "decision": {"promote_to_grpo": False},
        },
    )
    _write_jsonl(
        manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl",
        [
            {
                "task_id": "tasktrove_inferredbugs__a",
                "source": {"name": "tasktrove_inferredbugs", "policy": "train_allowed"},
                "environment": {"harness": "terminal_sandbox"},
                "splitting": {"split": "grpo_train"},
            },
            {
                "task_id": "tasktrove_inferredbugs__b",
                "source": {"name": "tasktrove_inferredbugs", "policy": "train_allowed"},
                "environment": {"harness": "terminal_sandbox"},
                "splitting": {"split": "grpo_train"},
            },
        ],
    )
    _write_jsonl(
        manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl",
        [
            {
                "task_id": "tasktrove_pymethods2test__a",
                "source": {"name": "tasktrove_pymethods2test", "policy": "train_allowed"},
                "environment": {"harness": "terminal_sandbox"},
                "splitting": {"split": "grpo_train"},
            }
        ],
    )
    _write_json(
        manifest_dir / "tasktrove_harbor" / "subset_selection.json",
        {
            "recommended_diversity_shards": [
                {"dataset": "laion/nemotron-gym-math-openmathreasoning", "role": "math"}
            ]
        },
    )

    completed_rows = [
        _rollout("tau-1", "tool_dialogue", "tau_bench_retail_train", "solo", "single_scaffold", 0.5),
        _rollout("tau-1", "tool_dialogue", "tau_bench_retail_train", "role", "role_workflow", 1.0),
        _rollout("repo-1", "repo_open_repo_terminal", "tasktrove_inferredbugs", "solo", "single_scaffold", 1.0),
        _rollout("repo-1", "repo_open_repo_terminal", "tasktrove_inferredbugs", "role", "role_workflow", 1.0),
    ]

    report = build_grpo_pilot_gap_plan(
        manifest_dir=manifest_dir,
        seed_jsonl=seed_jsonl,
        target_lane_counts={"tool_dialogue": 3, "repo_open_repo_terminal": 2, "long_context_memory_planning": 1},
        completed_rows=completed_rows,
        report_out=tmp_path / "gap.json",
    )

    assert report["status"] == "gap_plan_not_training_manifest"
    assert report["current_seed_task_count"] == 2
    assert report["lane_deficits"] == {
        "long_context_memory_planning": 1,
        "repo_open_repo_terminal": 1,
        "tool_dialogue": 2,
    }
    assert report["evidence_by_lane"]["tool_dialogue"]["reward_variance_groups"] == 1
    assert report["evidence_by_lane"]["tool_dialogue"]["role_improvement_groups"] == 1
    assert report["taskcraft_candidate_status"]["raw_dataset_grpo_ready"] is False
    assert report["taskcraft_candidate_status"]["audit_promote_to_grpo"] is False
    assert report["taskcraft_candidate_status"]["freeze_priority_count"] == 4
    assert report["taskcraft_candidate_status"]["audit_linkage_counts"]["any_atomic_match"] == 8
    assert report["tasktrove_reservoir"]["materialized_train_allowed_task_count"] == 3
    assert report["tasktrove_reservoir"]["materialized_train_allowed"]["tasktrove_inferredbugs"][
        "remaining_unseeded_count"
    ] == 1
    assert report["tasktrove_reservoir"]["recommended_diversity_shard_count"] == 1
    assert "TaskTrove" in report["lane_actions"]["repo_open_repo_terminal"]
    assert "TaskCraft" in report["lane_actions"]["long_context_memory_planning"]
    assert report["next_expansion_queue"][0]["source"] == "tau_bench_retail_train"
    assert report["next_expansion_queue"][0]["observed_variance_rate"] == 1.0
    assert report["deprioritized_sources"] == []
    assert "Do not start GRPO from the 2-task seed alone." in report["go_no_go"]


def test_gap_plan_recommends_counterfactual_long_context_when_reward_varying(tmp_path):
    manifest_dir = tmp_path / "manifest"
    seed_jsonl = manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl"
    _write_jsonl(seed_jsonl, [{"lane": "long_context_memory_planning", "source": "longctx_counterfactual"}])

    completed_rows = [
        _rollout("lc-1", "long_context_memory_planning", "longctx_counterfactual", "solo-a", "single_scaffold", 1.0),
        _rollout("lc-1", "long_context_memory_planning", "longctx_counterfactual", "solo-b", "single_scaffold", 0.5),
    ]

    report = build_grpo_pilot_gap_plan(
        manifest_dir=manifest_dir,
        seed_jsonl=seed_jsonl,
        target_lane_counts={"long_context_memory_planning": 3},
        completed_rows=completed_rows,
        report_out=tmp_path / "gap.json",
    )

    assert "counterfactual long-context" in report["lane_actions"]["long_context_memory_planning"]
    assert report["next_expansion_queue"][0]["source"] == "longctx_counterfactual"
    assert "OpenRouter-only" in report["next_expansion_queue"][0]["action"]
