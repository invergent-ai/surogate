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
        "tool_dialog_mimo_agent",
        "direct_gemini_synth",
        "direct_gpt_reasoner",
        "direct_opus_reviewer",
    }.issubset(names)
    assert {"codex", "claude_code", "opencode", "tool_dialog", "direct_qa"}.issubset(backends)
    assert [worker.worker_id for worker in workers] == list(range(len(workers)))
    harnesses = worker_harness_map(workers)
    assert harnesses["codex_gpt_coding_agent"] == "codex"
    assert harnesses["claude_code_opus_debugger"] == "claude_code"
    assert harnesses["opencode_kimi_builder"] == "opencode"


def test_scaffold_arms_are_valid_workflows():
    workers = canonical_workers()
    arms = canonical_arms(workers)
    assert len(arms) >= 20
    assert {arm.domain for arm in arms} == {"coding_repo", "tool_dialog", "direct_reasoning"}
    assert any(arm.name == "codex_build__claude_debug__codex_repair" for arm in arms)
    assert any(arm.name == "gpt_math__gemini_verify__opus_final" for arm in arms)

    for arm in arms:
        validate_workflow(arm.workflow, worker_count=len(workers))
        assert set(arm.worker_names)


def test_plan_counts_default_tournament():
    plan = build_plan()
    assert plan["task_mix"] == DEFAULT_TASK_MIX
    assert sum(plan["task_mix"].values()) == 37
    assert 20 <= sum(plan["task_mix"].values()) <= 40
    assert plan["live_calls"] is False
    assert plan["total_rollouts"] > sum(plan["task_mix"].values())
    assert plan["total_worker_calls"] > plan["total_rollouts"]
    assert "best individual model+scaffold worker on the same task set" in plan["fair_baselines"]
    assert plan["worker_harnesses"]["codex_gpt_coding_agent"] == "codex"


def test_plan_counts_custom_mix():
    plan = build_plan({"coding_repo": 2, "tool_dialog": 1, "direct_reasoning": 1})
    arms_by_domain = {}
    for arm in plan["arms"]:
        arms_by_domain.setdefault(arm["domain"], []).append(arm)

    assert plan["rollouts_by_domain"]["coding_repo"] == 2 * len(arms_by_domain["coding_repo"])
    assert plan["rollouts_by_domain"]["tool_dialog"] == len(arms_by_domain["tool_dialog"])
    assert plan["rollouts_by_domain"]["direct_reasoning"] == len(arms_by_domain["direct_reasoning"])


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


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
    return manifest_dir


def test_concrete_manifest_selects_tasks_and_jobs(tmp_path):
    manifest_dir = _fixture_manifest_dir(tmp_path)
    manifest = build_concrete_manifest(
        manifest_dir,
        task_mix={"coding_repo": 2, "tool_dialog": 2, "direct_reasoning": 3},
        seed=0,
    )
    assert manifest["deficits"] == {"coding_repo": 0, "tool_dialog": 0, "direct_reasoning": 0}
    assert manifest["selected_task_counts"] == {
        "coding_repo": 2,
        "tool_dialog": 2,
        "direct_reasoning": 3,
    }
    expected_jobs = (
        2 * len(manifest["arms_by_domain"]["coding_repo"])
        + 2 * len(manifest["arms_by_domain"]["tool_dialog"])
        + 3 * len(manifest["arms_by_domain"]["direct_reasoning"])
    )
    assert manifest["job_count"] == expected_jobs
    assert manifest["worker_call_count"] > manifest["job_count"]
    assert all("worker_harnesses" in job for job in manifest["jobs"])


def test_concrete_manifest_reports_coding_deficit_and_writes_jobs(tmp_path):
    manifest_dir = _fixture_manifest_dir(tmp_path)
    out = tmp_path / "manifest.json"
    jobs_out = tmp_path / "jobs.jsonl"
    manifest = write_concrete_manifest(
        manifest_dir,
        out,
        jobs_out,
        task_mix={"coding_repo": 3, "tool_dialog": 0, "direct_reasoning": 0},
        seed=0,
    )
    assert manifest["deficits"]["coding_repo"] == 1
    assert manifest["blocked_reasons"]
    assert out.exists()
    assert jobs_out.exists()
    assert len(jobs_out.read_text().splitlines()) == manifest["job_count"]


def test_concrete_manifest_fills_coding_from_local_deep_swe(tmp_path):
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
        task_mix={"coding_repo": 3, "tool_dialog": 0, "direct_reasoning": 0},
        seed=0,
    )
    assert manifest["deficits"]["coding_repo"] == 0
    assert manifest["selected_task_counts"]["coding_repo"] == 3
    assert [task["source"] for task in manifest["tasks"]].count("deep_swe_local") == 2


def test_readiness_report_splits_ready_and_pending_jobs(tmp_path):
    manifest_dir = _fixture_manifest_dir(tmp_path)
    out = tmp_path / "manifest.json"
    manifest = write_concrete_manifest(
        manifest_dir,
        out,
        None,
        task_mix={"coding_repo": 1, "tool_dialog": 2, "direct_reasoning": 3},
        seed=0,
    )
    report = analyze_readiness(out)
    expected_ready = manifest["selected_task_counts"]["direct_reasoning"] * len(
        manifest["arms_by_domain"]["direct_reasoning"]
    )
    expected_payload_pending = 3 * manifest["selected_task_counts"]["coding_repo"]
    assert report["jobs_by_status"]["ready"] == expected_ready
    assert report["jobs_by_status"]["payload_pending"] == expected_payload_pending
    assert report["jobs_by_status"]["adapter_pending"] == (
        manifest["selected_task_counts"]["coding_repo"] * (len(manifest["arms_by_domain"]["coding_repo"]) - 3)
    )
    assert report["jobs_by_status"]["harness_pending"] == (
        manifest["selected_task_counts"]["tool_dialog"] * len(manifest["arms_by_domain"]["tool_dialog"])
    )
    assert report["live_calls"] is False
