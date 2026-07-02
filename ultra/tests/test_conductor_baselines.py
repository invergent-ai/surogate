import json

from ultra.conductor_baselines import build_conductor_baseline_report, iter_discovery_shards, summarize_shard


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rollout(path, job_id, reward, success):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rollout_id": job_id,
                "reward": reward,
                "grade": {"score": 1.0 if success else 0.0, "success": success},
                "valid_for_training": True,
                "outcome_class": "valid_correct_trainable" if success else "valid_incorrect_trainable",
            }
        )
        + "\n"
    )


def _job(job_id, task_id, arm, stage):
    return {
        "job_id": job_id,
        "tournament_task_id": task_id,
        "lane": "tool_dialogue",
        "arm_domain": "tool_dialogue",
        "source": "tau_bench_retail_train",
        "source_task_id": task_id,
        "arm": arm,
        "stage": stage,
        "worker_names": ["worker"],
    }


def test_summarize_shard_scores_pre_rl_baseline_selectors(tmp_path):
    jobs_jsonl = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "runs"
    jobs = [
        _job("job_1", "task_a", "solo__tool_dialog_glm_agent", "single_scaffold"),
        _job("job_2", "task_a", "mimo_tool__opus_review", "role_workflow"),
        _job("job_3", "task_a", "glm_tool_attempt__mimo_repair", "role_workflow"),
        _job("job_4", "task_b", "solo__tool_dialog_glm_agent", "single_scaffold"),
        _job("job_5", "task_b", "mimo_tool__opus_review", "role_workflow"),
        _job("job_6", "task_b", "glm_tool_attempt__mimo_repair", "role_workflow"),
    ]
    _write_jsonl(jobs_jsonl, jobs)
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 0.5, False)
    _rollout(out_dir / "rollouts" / "job_2.json", "job_2", 0.5, False)
    _rollout(out_dir / "rollouts" / "job_3.json", "job_3", 1.0, True)
    _rollout(out_dir / "rollouts" / "job_4.json", "job_4", 1.0, True)
    _rollout(out_dir / "rollouts" / "job_5.json", "job_5", 0.5, False)
    _rollout(out_dir / "rollouts" / "job_6.json", "job_6", 1.0, True)

    report = summarize_shard("tau_test", jobs_jsonl, out_dir)

    assert report["task_groups"] == 2
    assert report["task_groups_with_reward_variance"] == 2
    assert report["prompt_only"]["successes"] == 0
    assert report["prompt_only"]["mean_reward"] == 0.5
    assert report["syntax_topology_sft"]["successes"] == 2
    assert report["syntax_topology_sft"]["mean_reward"] == 1.0
    assert report["best_single"]["successes"] == 1
    assert report["best_role"]["successes"] == 2


def test_load_rollout_rows_normalizes_tasktrove_unit_code_lane(tmp_path):
    from ultra.conductor_baselines import load_rollout_rows

    jobs_jsonl = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "runs"
    _write_jsonl(
        jobs_jsonl,
        [
            {
                **_job("job_1", "task_a", "solo__terminal_kimi_agent", "single_scaffold"),
                "lane": "repo_open_repo_terminal",
                "source": "tasktrove_pymethods2test",
            }
        ],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    rows = load_rollout_rows(jobs_jsonl, out_dir)

    assert rows[0]["lane"] == "unit_and_scientific_code"


def test_build_conductor_baseline_report_writes_outputs(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "tau_bench_retail_train"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "mimo_tool__opus_review", "role_workflow")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    report = build_conductor_baseline_report(
        manifest_dir=manifest_dir,
        report_out=json_out,
        md_out=md_out,
    )

    assert report["version"] == "fugu_ultra_conductor_baselines_v1"
    assert report["aggregate"]["prompt_only"]["successes"] == 1
    assert json_out.exists()
    assert "Pre-RL Conductor Baselines" in md_out.read_text()


def test_iter_discovery_shards_finds_tasktrove_prefilter_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "tasktrove_prefilter_batch_010"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning_open_singles"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__terminal_kimi_agent", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "tasktrove_prefilter_batch_010/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "tasktrove_prefilter_batch_010/scaffold_discovery_high_reasoning_open_singles"
        for shard in shards
    )


def test_iter_discovery_shards_skips_capped_diagnostic_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "tasktrove_prefilter_batch_011"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__terminal_kimi_agent", "single_scaffold")],
    )
    diagnostic_dirs = [
        shard / "scaffold_discovery_high_reasoning_open_singles_retry_capped1024_probe1",
        shard / "scaffold_discovery_high_reasoning_open_singles_retry_cap4096_p8_probe12",
    ]
    for out_dir in diagnostic_dirs:
        _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert not any("capped1024" in shard["out_dir"] for shard in shards)
    assert not any("cap4096" in shard["out_dir"] for shard in shards)


def test_iter_discovery_shards_skips_partial_do_not_promote_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "tasktrove_prefilter_agenttrove_exact_003"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning_open_singles"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__terminal_kimi_agent", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)
    (out_dir / "partial_run_summary.json").write_text(
        json.dumps(
            {
                "status": "partial_stopped_after_stalled_harbor_tail",
                "decision": "do_not_promote_source_to_grpo_without_verifier_audit",
            }
        )
        + "\n"
    )

    shards = iter_discovery_shards(manifest_dir)

    assert not any(
        shard["out_dir"]
        == "tasktrove_prefilter_agenttrove_exact_003/scaffold_discovery_high_reasoning_open_singles"
        for shard in shards
    )


def test_iter_discovery_shards_finds_tau_expansion_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "tau_bench_retail_high_action_002"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__tool_dialog_glm_agent", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "tau_bench_retail_high_action_002/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "tau_bench_retail_high_action_002/scaffold_discovery_high_reasoning"
        for shard in shards
    )


def test_iter_discovery_shards_finds_direct_unit_expansion_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "direct_unit_expansion_001"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__direct_gpt_reasoner", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "direct_unit_expansion_001/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "direct_unit_expansion_001/scaffold_discovery_high_reasoning"
        for shard in shards
    )


def test_iter_discovery_shards_finds_label_prior_expansion_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "label_prior_expansion_001"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning_nongpt_singles"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__direct_gemini_synth", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "label_prior_expansion_001/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "label_prior_expansion_001/scaffold_discovery_high_reasoning_nongpt_singles"
        for shard in shards
    )


def test_iter_discovery_shards_finds_label_prior_code_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "label_prior_code_flash_probe_002"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning_singles_nogpt"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__code_flash_fast", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "label_prior_code_flash_probe_002/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "label_prior_code_flash_probe_002/scaffold_discovery_high_reasoning_singles_nogpt"
        for shard in shards
    )


def test_iter_discovery_shards_finds_trace_branch_rollouts_with_global_jobs(tmp_path):
    manifest_dir = tmp_path / "manifest"
    jobs_jsonl = manifest_dir / "scaffold_tournament_jobs.jsonl"
    out_dir = manifest_dir / "trace_branch_open_singles_001" / "scaffold_discovery_high_reasoning_open_singles"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__opencode_kimi_builder", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "trace_branch_open_singles_001/scaffold_discovery_high_reasoning_open_singles"
        for shard in shards
    )


def test_iter_discovery_shards_finds_long_context_expansion_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "long_context_flash_probe_001"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning_flash_solo"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__long_flash_fast", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "long_context_flash_probe_001/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "long_context_flash_probe_001/scaffold_discovery_high_reasoning_flash_solo"
        for shard in shards
    )


def test_iter_discovery_shards_finds_expert_disagreement_rollouts(tmp_path):
    manifest_dir = tmp_path / "manifest"
    shard = manifest_dir / "expert_disagreement_v2"
    jobs_jsonl = shard / "scaffold_tournament_jobs.jsonl"
    out_dir = shard / "scaffold_discovery_high_reasoning"
    _write_jsonl(
        jobs_jsonl,
        [_job("job_1", "task_a", "solo__direct_gpt_reasoner", "single_scaffold")],
    )
    _rollout(out_dir / "rollouts" / "job_1.json", "job_1", 1.0, True)

    shards = iter_discovery_shards(manifest_dir)

    assert any(
        shard["jobs"] == "expert_disagreement_v2/scaffold_tournament_jobs.jsonl"
        and shard["out_dir"] == "expert_disagreement_v2/scaffold_discovery_high_reasoning"
        for shard in shards
    )
