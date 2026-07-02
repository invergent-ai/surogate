import json
import asyncio

import pytest

from ultra.scaffold_discovery_run import analyze_scaffold_discovery, run_scaffold_discovery_jobs


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _direct_task(task_id="direct-task-1"):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": "general",
        "source": {"name": "existing_bank", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": "Say alpha."}]},
        "environment": {"harness": "direct_qa"},
        "grader": {"type": "contains", "expected_answer": "alpha", "success_threshold": 1.0},
        "splitting": {
            "group_id": f"existing_bank/{task_id}",
            "split": "grpo_train",
            "contamination_group": f"existing_bank/{task_id}",
        },
        "metadata": {"domain": "general", "tags": ["test"]},
    }


def _job(task_jsonl, *, job_id="job_00001"):
    return {
        "job_id": job_id,
        "tournament_task_id": "math_science_knowledge::existing_bank::direct-task-1",
        "lane": "math_science_knowledge",
        "arm_domain": "math_science_knowledge",
        "source": "existing_bank",
        "source_task_id": "direct-task-1",
        "task_jsonl": str(task_jsonl),
        "task_harness": "direct_qa",
        "arm": "solo__direct_gpt_reasoner",
        "stage": "single_scaffold",
        "worker_names": ["direct_gpt_reasoner"],
        "worker_harnesses": ["direct_qa"],
        "worker_harness_map": {"direct_gpt_reasoner": "direct_qa"},
        "worker_calls": 1,
    }


def _tasktrove_unit_job(task_jsonl, *, job_id="job_00003"):
    row = _job(task_jsonl, job_id=job_id)
    row.update(
        {
            "tournament_task_id": "repo_open_repo_terminal::tasktrove_pymethods2test::pymethods2test-1",
            "lane": "repo_open_repo_terminal",
            "arm_domain": "terminal_sandbox",
            "source": "tasktrove_pymethods2test",
            "source_task_id": "direct-task-1",
            "arm": "solo__terminal_kimi_agent",
        }
    )
    return row


def _role_job(task_jsonl, *, job_id="job_00002"):
    row = _job(task_jsonl, job_id=job_id)
    row.update(
        {
            "arm": "flash_answer__gemini_audit",
            "stage": "role_workflow",
            "worker_names": ["direct_flash_fast", "direct_gemini_synth"],
            "worker_harnesses": ["direct_qa", "direct_qa"],
            "worker_harness_map": {"direct_flash_fast": "direct_qa", "direct_gemini_synth": "direct_qa"},
            "worker_calls": 2,
        }
    )
    return row


@pytest.mark.asyncio
async def test_scaffold_discovery_run_dry_run_selects_jobs(tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    _write_jsonl(tasks_jsonl, [_direct_task()])
    _write_jsonl(jobs_jsonl, [_job(tasks_jsonl), _job(tasks_jsonl, job_id="job_00002")])

    report = await run_scaffold_discovery_jobs(
        jobs_jsonl=jobs_jsonl,
        out_dir=tmp_path / "runs",
        report_out=tmp_path / "dry.json",
        dry_run=True,
        lanes={"math_science_knowledge"},
        limit=1,
        dotenv=tmp_path / "missing.env",
    )

    assert report["mode"] == "dry_run"
    assert report["live_calls"] is False
    assert report["selected_jobs"] == 1
    assert report["selected_sample"][0]["job_id"] == "job_00001"
    assert json.loads((tmp_path / "dry.json").read_text())["selected_jobs"] == 1


@pytest.mark.asyncio
async def test_scaffold_discovery_rejects_openrouter_override_for_gpt_jobs(tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    _write_jsonl(tasks_jsonl, [_direct_task()])
    _write_jsonl(jobs_jsonl, [_job(tasks_jsonl)])

    with pytest.raises(ValueError, match="GPT workers must not be routed through OpenRouter"):
        await run_scaffold_discovery_jobs(
            jobs_jsonl=jobs_jsonl,
            out_dir=tmp_path / "runs",
            report_out=tmp_path / "dry.json",
            dry_run=True,
            provider_name="openrouter",
            dotenv=tmp_path / "missing.env",
        )


@pytest.mark.asyncio
async def test_scaffold_discovery_runs_terminal_docker_janitor_preflight(monkeypatch, tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    terminal_job = _job(tasks_jsonl)
    terminal_job["task_harness"] = "terminal_sandbox"
    _write_jsonl(tasks_jsonl, [_direct_task()])
    _write_jsonl(jobs_jsonl, [terminal_job])
    calls = []

    def fake_janitor(**kwargs):
        calls.append(kwargs)
        return {"mode": "dry_run", "stale_candidates": 3}

    monkeypatch.setattr("ultra.scaffold_discovery_run.cleanup_stale_docker_networks", fake_janitor)

    report = await run_scaffold_discovery_jobs(
        jobs_jsonl=jobs_jsonl,
        out_dir=tmp_path / "runs",
        report_out=tmp_path / "dry.json",
        dry_run=True,
        docker_network_janitor=True,
        dotenv=tmp_path / "missing.env",
    )

    assert calls == [{"dry_run": True}]
    assert report["docker_network_janitor"] == {"mode": "dry_run", "stale_candidates": 3}


@pytest.mark.asyncio
async def test_scaffold_discovery_run_fake_executes_and_analyzes(tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "runs"
    _write_jsonl(tasks_jsonl, [_direct_task()])
    _write_jsonl(jobs_jsonl, [_job(tasks_jsonl), _role_job(tasks_jsonl)])

    report = await run_scaffold_discovery_jobs(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "run.json",
        fake=True,
        limit=2,
        dotenv=tmp_path / "missing.env",
    )

    assert report["mode"] == "fake"
    assert report["counts"] == {"ok": 2}
    assert report["rows"][0]["reward"] == 0.5
    assert (out_dir / "rollouts" / "job_00001.json").exists()
    assert (out_dir / "rollouts" / "job_00002.json").exists()

    analysis = analyze_scaffold_discovery(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "analysis.json",
    )
    assert analysis["rollouts"] == 2
    assert analysis["by_lane"]["math_science_knowledge"]["successes"] == 0
    assert analysis["paired_single_vs_role"]["task_groups"] == 1
    assert analysis["paired_single_vs_role"]["role_matches_best_single_reward"] == 1
    assert analysis["go_no_go_hint"] == "no_variance_observed"


@pytest.mark.asyncio
async def test_scaffold_discovery_analysis_normalizes_tasktrove_unit_lane(tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "runs"
    _write_jsonl(tasks_jsonl, [_direct_task()])
    _write_jsonl(jobs_jsonl, [_tasktrove_unit_job(tasks_jsonl)])

    await run_scaffold_discovery_jobs(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "run.json",
        fake=True,
        dotenv=tmp_path / "missing.env",
    )

    analysis = analyze_scaffold_discovery(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "analysis.json",
    )

    assert "unit_and_scientific_code" in analysis["by_lane"]
    assert "repo_open_repo_terminal" not in analysis["by_lane"]


@pytest.mark.asyncio
async def test_scaffold_discovery_pool_only_rollouts_are_not_trainable(tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "runs"
    task = _direct_task()
    task["source"]["policy"] = "pool_only"
    task["splitting"]["split"] = "pool_validation"
    job = _job(tasks_jsonl)
    job["source_policy"] = "pool_only"
    job["task_split"] = "pool_validation"
    _write_jsonl(tasks_jsonl, [task])
    _write_jsonl(jobs_jsonl, [job])

    report = await run_scaffold_discovery_jobs(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "run.json",
        fake=True,
        dotenv=tmp_path / "missing.env",
    )

    assert report["rows"][0]["valid_for_training"] is False
    rollout = json.loads((out_dir / "rollouts" / "job_00001.json").read_text())
    assert rollout["valid_for_training"] is False

    analysis = analyze_scaffold_discovery(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "analysis.json",
    )
    assert analysis["by_lane"]["math_science_knowledge"]["trainable"] == 0


@pytest.mark.asyncio
async def test_scaffold_discovery_run_job_timeout(monkeypatch, tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "runs"
    _write_jsonl(tasks_jsonl, [_direct_task()])
    _write_jsonl(jobs_jsonl, [_job(tasks_jsonl)])

    async def slow_run_one(*args, **kwargs):
        await asyncio.sleep(0.05)
        return "ok", {"job_id": "job_00001"}

    monkeypatch.setattr("ultra.scaffold_discovery_run._run_one", slow_run_one)

    report = await run_scaffold_discovery_jobs(
        jobs_jsonl=jobs_jsonl,
        out_dir=out_dir,
        report_out=tmp_path / "timeout.json",
        fake=True,
        job_timeout_s=0.001,
        dotenv=tmp_path / "missing.env",
    )

    assert report["counts"] == {"timeout": 1}
    assert report["rows"][0]["status"] == "timeout"
    assert (out_dir / "errors" / "job_00001.json").exists()
