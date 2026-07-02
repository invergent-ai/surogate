import json

from ultra.scaffold_tournament import (
    DEFAULT_TASK_MIX,
    analyze_readiness,
    build_concrete_manifest,
    build_plan,
    canonical_arms,
    canonical_workers,
    write_concrete_manifest,
    worker_harness_map,
)
from ultra.workflow import validate_workflow


def test_canonical_workers_are_scaffold_aware():
    workers = canonical_workers()
    names = {worker.name for worker in workers}
    backends = {worker.backend for worker in workers}

    assert {
        "codex_gpt_coding_agent",
        "claude_code_opus_debugger",
        "opencode_kimi_builder",
        "opencode_mimo_repair",
        "opencode_glm_builder",
        "opencode_flash_challenger",
        "opencode_minimax_challenger",
        "opencode_deepseek_pro_challenger",
        "terminal_gpt_agent",
        "terminal_kimi_agent",
        "terminal_glm_agent",
        "tool_dialog_mimo_agent",
        "direct_gemini_synth",
        "direct_gpt_reasoner",
        "direct_opus_reviewer",
        "direct_glm_reasoner",
        "direct_mimo_reasoner",
        "direct_minimax_reasoner",
    }.issubset(names)
    assert {"codex", "claude_code", "opencode", "tool_dialog", "direct_qa", "terminal"}.issubset(backends)
    assert [worker.worker_id for worker in workers] == list(range(len(workers)))
    harnesses = worker_harness_map(workers)
    assert harnesses["codex_gpt_coding_agent"] == "codex"
    assert harnesses["claude_code_opus_debugger"] == "claude_code"
    assert harnesses["opencode_kimi_builder"] == "opencode"
    assert harnesses["opencode_flash_challenger"] == "opencode"
    assert harnesses["terminal_gpt_agent"] == "terminal_sandbox"


def test_scaffold_arms_are_valid_workflows():
    workers = canonical_workers()
    arms = canonical_arms(workers)
    assert len(arms) >= 20
    assert {arm.domain for arm in arms} == {
        "repo_coding",
        "terminal_sandbox",
        "unit_and_scientific_code",
        "tool_dialogue",
        "math_science_knowledge",
        "long_context_memory_planning",
    }
    assert any(arm.name == "codex_build__claude_debug__codex_repair" for arm in arms)
    assert any(arm.name == "gpt_math__gemini_verify__opus_final" for arm in arms)
    assert any(arm.name == "terminal_gpt_plan__kimi_solve" for arm in arms)
    assert any(arm.name == "solo__terminal_glm_agent" for arm in arms)
    assert any(arm.name == "terminal_kimi_attempt__glm_repair" for arm in arms)
    assert any(arm.name == "solo__code_glm_reasoner" for arm in arms)
    assert any(arm.name == "solo__code_mimo_reasoner" for arm in arms)
    assert any(arm.name == "solo__code_minimax_reasoner" for arm in arms)
    assert any(arm.name == "solo__direct_glm_reasoner" for arm in arms)
    assert any(arm.name == "solo__long_glm_reasoner" for arm in arms)
    assert any(arm.name == "solo__long_mimo_reasoner" for arm in arms)
    assert any(arm.name == "solo__long_minimax_reasoner" for arm in arms)
    assert any(arm.name == "solo__code_flash_fast" for arm in arms)
    assert any(arm.name == "solo__long_flash_fast" for arm in arms)
    assert any(arm.name == "solo__opencode_flash_challenger" for arm in arms)
    assert any(arm.name == "solo__opencode_minimax_challenger" for arm in arms)
    assert any(arm.name == "solo__opencode_deepseek_pro_challenger" for arm in arms)
    assert any(arm.name == "long_gpt_extract__gemini_verify__opus_final" for arm in arms)

    for arm in arms:
        validate_workflow(arm.workflow, worker_count=len(workers))
        assert set(arm.worker_names)


def test_plan_counts_default_tournament():
    plan = build_plan()
    assert plan["task_mix"] == DEFAULT_TASK_MIX
    assert sum(plan["task_mix"].values()) == 200
    assert plan["live_calls"] is False
    assert plan["total_rollouts"] > sum(plan["task_mix"].values())
    assert plan["total_worker_calls"] > plan["total_rollouts"]
    assert "best individual model+scaffold worker on the same task set" in plan["fair_baselines"]
    assert plan["worker_harnesses"]["codex_gpt_coding_agent"] == "codex"


def test_plan_counts_custom_mix():
    plan = build_plan(
        {
            "repo_open_repo_terminal": 2,
            "unit_and_scientific_code": 1,
            "math_science_knowledge": 1,
            "tool_dialogue": 1,
            "long_context_memory_planning": 1,
        }
    )
    arms_by_domain = {}
    for arm in plan["arms"]:
        arms_by_domain.setdefault(arm["domain"], []).append(arm)

    repo_arm_count = len(arms_by_domain["repo_coding"]) + len(arms_by_domain["terminal_sandbox"])
    assert plan["rollouts_by_domain"]["repo_open_repo_terminal"] == 2 * repo_arm_count
    assert plan["rollouts_by_domain"]["unit_and_scientific_code"] == len(
        arms_by_domain["unit_and_scientific_code"]
    )
    assert plan["rollouts_by_domain"]["tool_dialogue"] == len(arms_by_domain["tool_dialogue"])
    assert plan["rollouts_by_domain"]["math_science_knowledge"] == len(arms_by_domain["math_science_knowledge"])


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _repo_taskspec(task_id, source_name, *, tags=None):
    assets = [
        {
            "opencode_instance": {
                "image_name": f"example/{task_id}:latest",
                "problem_statement": "Fix it.",
                "testbed": "/app",
            }
        }
    ]
    if source_name == "trace_state_branches":
        assets.append(
            {
                "trace_branch": {
                    "origin_harness": "codex",
                    "previous_success": False,
                    "worker_model": "gpt-5.5",
                }
            }
        )
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "agentic_coding",
        "source": {"name": source_name, "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Fix it"}], "assets": assets},
        "environment": {"harness": "opencode", "wall_time_seconds": 900},
        "grader": {"type": "deep_swe_hidden_tests", "success_threshold": 1.0},
        "splitting": {
            "group_id": f"{source_name}/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"{source_name}/{task_id}",
        },
        "metadata": {"domain": "software_engineering", "tags": tags or []},
    }


def _terminal_taskspec(tmp_path, task_id, source_name="tasktrove_pymethods2test"):
    task_dir = tmp_path / "harbor" / task_id
    task_dir.mkdir(parents=True)
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "terminal_tool_use",
        "source": {"name": source_name, "version": "v1", "policy": "train_allowed"},
        "input": {
            "messages": [{"role": "user", "content": "Solve with terminal tools."}],
            "assets": [{"harbor_task": {"task_dir": str(task_dir), "agent": "terminus-2"}}],
        },
        "environment": {"harness": "terminal_sandbox", "wall_time_seconds": 900},
        "grader": {"type": "container_command", "success_threshold": 1.0},
        "splitting": {
            "group_id": f"{source_name}/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"{source_name}/{task_id}",
        },
        "metadata": {"domain": "software_engineering", "subdomain": "terminal", "tags": ["harbor"]},
    }


def _code_taskspec(task_id):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "code",
        "source": {"name": "existing_bank", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Write a function."}]},
        "environment": {"harness": "code_exec"},
        "grader": {"type": "code_exec", "expected_answer": {"tests": []}, "success_threshold": 1.0},
        "splitting": {
            "group_id": f"existing_bank/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"existing_bank/{task_id}",
        },
        "metadata": {"domain": "code", "tags": ["code_exec"]},
    }


def _direct_taskspec(task_id, domain):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": domain,
        "source": {"name": "existing_bank", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Answer."}]},
        "environment": {"harness": "direct_qa"},
        "grader": {"type": "exact_match", "expected_answer": "A", "success_threshold": 1.0},
        "splitting": {
            "group_id": f"existing_bank/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"existing_bank/{task_id}",
        },
        "metadata": {"domain": domain, "tags": ["direct"]},
    }


def _tool_taskspec(task_id, subdomain):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "tool_dialogue",
        "source": {"name": "tau_custom", "version": "v1", "policy": "train_allowed"},
        "input": {
            "messages": [{"role": "user", "content": "Update the state."}],
            "tools": [{"type": "function", "function": {"name": "finish", "parameters": {}}}],
        },
        "environment": {"harness": "tool_dialog"},
        "grader": {
            "type": "tool_dialog_state",
            "expected_answer": {"initial_state": {}, "success": [{"path": ["done"], "equals": True}]},
            "success_threshold": 1.0,
        },
        "splitting": {
            "group_id": f"tau_custom/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"tau_custom/{task_id}",
        },
        "metadata": {"domain": "tool_dialogue", "subdomain": subdomain, "tags": ["tau"]},
    }


def _long_taskspec(task_id):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "long_context",
        "source": {"name": "longctx_generated", "version": "v1", "policy": "train_allowed"},
        "input": {
            "messages": [{"role": "user", "content": "What is the code?"}],
            "context_documents": [{"title": "doc", "text": "The code is A."}],
        },
        "environment": {"harness": "long_context"},
        "grader": {"type": "exact_match", "expected_answer": "A", "success_threshold": 1.0},
        "splitting": {
            "group_id": f"longctx/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"longctx/{task_id}",
        },
        "metadata": {"domain": "long_context", "requires_long_context": True, "tags": ["longctx"]},
    }


def _fixture_manifest_dir(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    manifest_dir.mkdir()
    (manifest_dir / "agentic_coding_frontier_direct3.plan.json").write_text(
        json.dumps({"tasks": ["repo-task-0", "repo-task-1"]})
    )
    _write_jsonl(
        manifest_dir / "agentic_coding_frontier_direct3.jsonl",
        [
            {"task_id": "repo-task-0", "workers": ["kimi-code"], "reward": 1.0, "valid": True},
            {"task_id": "repo-task-0", "workers": ["opus"], "reward": 0.0, "valid": True},
            {"task_id": "repo-task-1", "workers": ["mimo"], "reward": 1.0, "valid": True},
            {"task_id": "repo-task-1", "workers": ["glm"], "reward": 0.0, "valid": True},
        ],
    )
    tau_rows = []
    for domain in ["tau_retail", "tau_airline"]:
        for i in range(2):
            for worker, reward in {"mimo": 1.0, "glm": float(i), "deepseek_flash": 0.0}.items():
                tau_rows.append(
                    {
                        "domain": domain,
                        "item_id": f"{domain}-{i}",
                        "worker": worker,
                        "reward": reward,
                    }
                )
    _write_jsonl(manifest_dir / "agentic_bank.jsonl", tau_rows)

    direct_rows = []
    for domain in ["math", "science", "general"]:
        for i in range(2):
            direct_rows.append(
                {
                    "task_id": f"{domain}-{i}",
                    "domain": domain,
                    "source": f"{domain}_source",
                    "prompt": "Q",
                    "solution": "A",
                    "grader": "mc_letter",
                    "system": "",
                    "split": "test",
                    "verdict": "discriminative",
                    "rewards": [1, 0, 1, 0, 0, i],
                }
            )
    _write_jsonl(manifest_dir / "manifest.jsonl", direct_rows)
    mvp_rows = [
        _repo_taskspec(f"repo-{i}", "generated_repo_tasks", tags=["generated_repo"]) for i in range(2)
    ]
    mvp_rows.extend(_terminal_taskspec(tmp_path, f"terminal-{i}") for i in range(3))
    mvp_rows.extend(_code_taskspec(f"code-{i}") for i in range(3))
    for domain in ["math", "science", "general"]:
        mvp_rows.extend(_direct_taskspec(f"{domain}-{i}", domain) for i in range(2))
    for subdomain in ["retail", "airline", "banking"]:
        mvp_rows.extend(_tool_taskspec(f"{subdomain}-{i}", subdomain) for i in range(2))
    mvp_rows.extend(_long_taskspec(f"long-{i}") for i in range(3))
    _write_jsonl(manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl", mvp_rows)
    return manifest_dir


def test_concrete_manifest_selects_tasks_and_jobs(tmp_path):
    manifest_dir = _fixture_manifest_dir(tmp_path)
    task_mix = {
        "repo_open_repo_terminal": 2,
        "unit_and_scientific_code": 2,
        "math_science_knowledge": 3,
        "tool_dialogue": 3,
        "long_context_memory_planning": 2,
    }
    manifest = build_concrete_manifest(
        manifest_dir,
        task_mix=task_mix,
        seed=0,
    )
    assert manifest["deficits"] == {lane: 0 for lane in task_mix}
    assert manifest["selected_task_counts"] == task_mix
    expected_jobs = sum(
        count * len(manifest["arms_by_domain"][domain])
        for domain, count in manifest["selected_arm_domain_counts"].items()
    )
    assert manifest["job_count"] == expected_jobs
    assert manifest["worker_call_count"] > manifest["job_count"]
    assert all("worker_harnesses" in job for job in manifest["jobs"])
    assert all("worker_harness_map" in job for job in manifest["jobs"])
    assert all(job["source_policy"] == "train_allowed" for job in manifest["jobs"])
    assert all(job["task_split"] == "grpo_train" for job in manifest["jobs"])


def test_tasktrove_unit_code_sources_use_unit_lane_with_terminal_arms(tmp_path):
    manifest_dir = tmp_path / "fugu_clean_v1"
    rows = [_terminal_taskspec(tmp_path, "pymethods2test-0001", source_name="tasktrove_pymethods2test")]
    _write_jsonl(manifest_dir / "tasktrove_unit.jsonl", rows)
    _write_jsonl(manifest_dir / "empty_branch_tasks.jsonl", [])

    manifest = build_concrete_manifest(
        manifest_dir,
        task_mix={
            "repo_open_repo_terminal": 0,
            "unit_and_scientific_code": 1,
            "math_science_knowledge": 0,
            "tool_dialogue": 0,
            "long_context_memory_planning": 0,
        },
        tasks_jsonl=manifest_dir / "tasktrove_unit.jsonl",
        branch_tasks_jsonl=manifest_dir / "empty_branch_tasks.jsonl",
        seed=0,
    )

    assert manifest["selected_task_counts"] == {"unit_and_scientific_code": 1}
    assert manifest["selected_arm_domain_counts"] == {"terminal_sandbox": 1}
    assert {job["lane"] for job in manifest["jobs"]} == {"unit_and_scientific_code"}
    assert {job["arm_domain"] for job in manifest["jobs"]} == {"terminal_sandbox"}
    assert all(job["task_harness"] == "terminal_sandbox" for job in manifest["jobs"])


def test_concrete_manifest_reports_coding_deficit_and_writes_jobs(tmp_path):
    manifest_dir = _fixture_manifest_dir(tmp_path)
    out = tmp_path / "manifest.json"
    jobs_out = tmp_path / "jobs.jsonl"
    manifest = write_concrete_manifest(
        manifest_dir,
        out,
        jobs_out,
        task_mix={
            "repo_open_repo_terminal": 6,
            "unit_and_scientific_code": 0,
            "math_science_knowledge": 0,
            "tool_dialogue": 0,
            "long_context_memory_planning": 0,
        },
        seed=0,
    )
    assert manifest["deficits"]["repo_open_repo_terminal"] == 4
    assert manifest["blocked_reasons"]
    assert out.exists()
    assert jobs_out.exists()
    assert len(jobs_out.read_text().splitlines()) == manifest["job_count"]


def test_concrete_manifest_prefers_train_allowed_repo_and_trace_tasks(tmp_path):
    manifest_dir = tmp_path / "director" / "manifests" / "fugu_clean_v1"
    manifest_dir.mkdir(parents=True)
    _write_jsonl(manifest_dir / "agentic_coding_frontier_direct3.jsonl", [])
    _write_jsonl(manifest_dir / "agentic_bank.jsonl", [])
    _write_jsonl(manifest_dir / "manifest.jsonl", [])
    _write_jsonl(
        manifest_dir / "trace_capture" / "branch_taskspecs.jsonl",
        [_repo_taskspec(f"trace-{i}", "trace_state_branches", tags=["trace_state_branch"]) for i in range(4)],
    )
    _write_jsonl(
        manifest_dir / "generated_repo_tasks" / "taskspecs.jsonl",
        [_repo_taskspec(f"repo-{i}", "generated_repo_tasks", tags=["generated_repo"]) for i in range(2)],
    )
    _write_jsonl(
        manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl",
        [_repo_taskspec(f"repo-{i}", "generated_repo_tasks", tags=["generated_repo"]) for i in range(2)],
    )

    manifest = build_concrete_manifest(
        manifest_dir,
        task_mix={
            "repo_open_repo_terminal": 2,
            "unit_and_scientific_code": 0,
            "math_science_knowledge": 0,
            "tool_dialogue": 0,
            "long_context_memory_planning": 0,
        },
        seed=0,
    )

    sources = [task["source"] for task in manifest["tasks"]]
    assert manifest["deficits"]["repo_open_repo_terminal"] == 0
    assert manifest["branch_shard_count"] == 4
    assert sources.count("trace_state_branches") == 4
    assert "generated_repo_tasks" in sources
    assert all("task_jsonl" in task for task in manifest["tasks"])
    assert "deep_swe_local" not in sources

    out = tmp_path / "train_manifest.json"
    written = write_concrete_manifest(
        manifest_dir,
        out,
        None,
        task_mix={
            "repo_open_repo_terminal": 2,
            "unit_and_scientific_code": 0,
            "math_science_knowledge": 0,
            "tool_dialogue": 0,
            "long_context_memory_planning": 0,
        },
        seed=0,
    )
    readiness = analyze_readiness(out)
    assert readiness["jobs_by_status"] == {"ready": written["job_count"]}


def test_concrete_manifest_pool_validation_requires_explicit_flag(tmp_path):
    manifest_dir = tmp_path / "director" / "manifests" / "fugu_clean_v1"
    manifest_dir.mkdir(parents=True)
    pool_task = _repo_taskspec("django__django-11292", "acrouter_swebench_verified")
    pool_task["source"]["policy"] = "pool_only"
    pool_task["splitting"]["split"] = "pool_validation"
    pool_task["grader"]["type"] = "swebench_verified_hidden_tests"
    pool_tasks = tmp_path / "pool_tasks.jsonl"
    _write_jsonl(pool_tasks, [pool_task])

    task_mix = {
        "repo_open_repo_terminal": 1,
        "unit_and_scientific_code": 0,
        "math_science_knowledge": 0,
        "tool_dialogue": 0,
        "long_context_memory_planning": 0,
    }
    default_manifest = build_concrete_manifest(
        manifest_dir,
        tasks_jsonl=pool_tasks,
        task_mix=task_mix,
        seed=0,
    )
    assert default_manifest["selected_task_counts"].get("repo_open_repo_terminal", 0) == 0
    assert default_manifest["deficits"]["repo_open_repo_terminal"] == 1

    pool_manifest = build_concrete_manifest(
        manifest_dir,
        tasks_jsonl=pool_tasks,
        task_mix=task_mix,
        seed=0,
        include_pool_validation=True,
    )
    assert pool_manifest["include_pool_validation"] is True
    assert pool_manifest["deficits"]["repo_open_repo_terminal"] == 0
    assert pool_manifest["selected_task_counts"]["repo_open_repo_terminal"] == 1
    assert pool_manifest["tasks"][0]["source"] == "acrouter_swebench_verified"
    assert pool_manifest["tasks"][0]["split"] == "pool_validation"
    assert "pool_only" in pool_manifest["tasks"][0]["selection_tags"]
    assert "train_allowed" not in pool_manifest["tasks"][0]["selection_tags"]
    assert all(job["source_policy"] == "pool_only" for job in pool_manifest["jobs"])
    assert all(job["task_split"] == "pool_validation" for job in pool_manifest["jobs"])


def test_concrete_manifest_does_not_fill_coding_from_local_deep_swe(tmp_path):
    manifest_dir = tmp_path / "director" / "manifests" / "fugu_clean_v1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "agentic_coding_frontier_direct3.plan.json").write_text(
        json.dumps({"tasks": ["repo-task-0"]})
    )
    _write_jsonl(
        manifest_dir / "agentic_coding_frontier_direct3.jsonl",
        [{"task_id": "repo-task-0", "workers": ["kimi-code"], "reward": 1.0, "valid": True}],
    )
    _write_jsonl(manifest_dir / "agentic_bank.jsonl", [])
    _write_jsonl(manifest_dir / "manifest.jsonl", [])

    tasks_root = tmp_path / "director" / "vendor" / "deep_swe" / "tasks"
    for i, language in enumerate(["python", "go", "typescript"]):
        task_dir = tasks_root / f"local-task-{i}"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text(f"Fix local task {i}.\n")
        (task_dir / "task.toml").write_text(
            f"""
schema_version = "1.1"
[metadata]
task_id = "local-task-{i}"
display_title = "Local task {i}"
category = "bugfix"
language = "{language}"
repository_url = "https://example.com/repo-{i}"
base_commit_hash = "commit-{i}"
"""
        )

    manifest = build_concrete_manifest(
        manifest_dir,
        task_mix={
            "repo_open_repo_terminal": 3,
            "unit_and_scientific_code": 0,
            "math_science_knowledge": 0,
            "tool_dialogue": 0,
            "long_context_memory_planning": 0,
        },
        seed=0,
    )
    assert manifest["deficits"]["repo_open_repo_terminal"] == 3
    assert manifest["selected_task_counts"].get("repo_open_repo_terminal", 0) == 0
    assert "Deep SWE remains final-eval-only" in manifest["blocked_reasons"][0]
    assert "deep_swe_local" not in [task["source"] for task in manifest["tasks"]]


def test_readiness_report_splits_ready_and_pending_jobs(tmp_path):
    manifest_dir = _fixture_manifest_dir(tmp_path)
    out = tmp_path / "manifest.json"
    manifest = write_concrete_manifest(
        manifest_dir,
        out,
        None,
        task_mix={
            "repo_open_repo_terminal": 2,
            "unit_and_scientific_code": 1,
            "math_science_knowledge": 3,
            "tool_dialogue": 2,
            "long_context_memory_planning": 1,
        },
        seed=0,
    )
    report = analyze_readiness(out)
    assert report["jobs_by_status"]["ready"] == manifest["job_count"]
    assert "adapter_pending" not in report["jobs_by_status"]
    assert "harness_pending" not in report["jobs_by_status"]
    assert "payload_pending" not in report["jobs_by_status"]
    assert report["live_calls"] is False
