import json
import asyncio

import pytest

from ultra.schemas import (
    EnvironmentSpec,
    Execution,
    ExecStep,
    Grade,
    GraderSpec,
    RepoRef,
    RolloutRecord,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
    Workflow,
    WorkflowStep,
)
from ultra.trace_capture_run import run_trace_capture_jobs


def _task(task_id="generated_repo_tasks__x"):
    return TaskSpec(
        task_id=task_id,
        capability="agentic_coding",
        source=SourceRef(name="generated_repo_tasks", version="v1", policy="train_allowed"),
        input=TaskInput(
            messages=[{"role": "user", "content": "Fix it"}],
            repo=RepoRef(url=f"local://{task_id}", base_commit="generated-v1"),
        ),
        environment=EnvironmentSpec(harness="opencode"),
        grader=GraderSpec(type="deep_swe_hidden_tests"),
        splitting=SplittingSpec(group_id="g", split="grpo_train"),
    )


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.asyncio
async def test_trace_capture_run_writes_rollout_trace_and_report(monkeypatch, tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks_jsonl, [_task().model_dump(mode="json")])
    jobs_jsonl = tmp_path / "jobs.jsonl"
    job = {
        "job_id": "j1",
        "task_id": "generated_repo_tasks__x",
        "task_jsonl": str(tasks_jsonl),
        "arm": "solo__opencode_kimi_builder",
        "budget": "short",
        "rollout_out": str(tmp_path / "rollouts" / "j1.json"),
        "agent_trace_out": str(tmp_path / "traces" / "j1.json"),
        "artifact_dir": str(tmp_path / "artifacts" / "j1"),
    }
    _write_jsonl(jobs_jsonl, [job])

    async def fake_run_canary(**kwargs):
        artifact_dir = kwargs["artifact_dir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        patch = artifact_dir / "patch.diff"
        prompt = artifact_dir / "prompt.txt"
        command = artifact_dir / "command.json"
        workspace = artifact_dir / "workspace_snapshot"
        grade_ref = artifact_dir / "grade.json"
        patch.write_text("diff --git a/x b/x\n")
        prompt.write_text("Fix it")
        command.write_text("{}")
        workspace.mkdir()
        grade_ref.write_text("{}")
        return RolloutRecord(
            rollout_id=kwargs["rollout_id"],
            task_id="generated_repo_tasks__x",
            source_name="generated_repo_tasks",
            capability="agentic_coding",
            harness="opencode",
            workflow=Workflow(steps=[WorkflowStep(worker_id=2, subtask="fix")]),
            execution=Execution(
                steps=[
                    ExecStep(
                        worker_id=2,
                        harness="opencode",
                        session_ref=str(workspace),
                        patch_ref=str(patch),
                        messages_ref=str(prompt),
                        tool_events_ref=str(command),
                        text=patch.read_text(),
                    )
                ]
            ),
            grade=Grade(score=1.0, success=True, grader_ref=str(grade_ref)),
            reward=1.0,
        )

    monkeypatch.setattr("ultra.trace_capture_run.run_canary", fake_run_canary)

    report = await run_trace_capture_jobs(
        jobs_jsonl=jobs_jsonl,
        report_out=tmp_path / "report.json",
        limit=1,
    )

    assert report["counts"] == {"ok": 1}
    assert report["parallel"] == 1
    assert report["skipped_existing"] == 0
    assert report["rows"][0]["trace_has_patch"] is True
    assert report["rows"][0]["trace_has_workspace"] is True
    assert (tmp_path / "rollouts" / "j1.json").exists()
    trace = json.loads((tmp_path / "traces" / "j1.json").read_text())
    assert trace["artifacts"]["final_patch_ref"]
    assert trace["artifacts"]["workspace_snapshot_ref"]

    resumed = await run_trace_capture_jobs(
        jobs_jsonl=jobs_jsonl,
        report_out=tmp_path / "report2.json",
        limit=1,
    )
    assert resumed["selected_jobs"] == 0
    assert resumed["skipped_existing"] == 1


@pytest.mark.asyncio
async def test_trace_capture_run_loads_dotenv(monkeypatch, tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks_jsonl, [_task().model_dump(mode="json")])
    jobs_jsonl = tmp_path / "jobs.jsonl"
    _write_jsonl(jobs_jsonl, [])
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENROUTER_API_KEY=test-key\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    report = await run_trace_capture_jobs(
        jobs_jsonl=jobs_jsonl,
        report_out=tmp_path / "report.json",
        dotenv=dotenv,
    )

    assert report["loaded_env_keys"] == ["OPENROUTER_API_KEY"]


@pytest.mark.asyncio
async def test_trace_capture_run_parallelizes_selected_jobs(monkeypatch, tmp_path):
    tasks_jsonl = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks_jsonl, [_task().model_dump(mode="json")])
    jobs = []
    for job_id in ["j1", "j2"]:
        jobs.append(
            {
                "job_id": job_id,
                "task_id": "generated_repo_tasks__x",
                "task_jsonl": str(tasks_jsonl),
                "arm": "solo__opencode_kimi_builder",
                "budget": "short",
                "rollout_out": str(tmp_path / "rollouts" / f"{job_id}.json"),
                "agent_trace_out": str(tmp_path / "traces" / f"{job_id}.json"),
                "artifact_dir": str(tmp_path / "artifacts" / job_id),
            }
        )
    jobs_jsonl = tmp_path / "jobs.jsonl"
    _write_jsonl(jobs_jsonl, jobs)

    active = 0
    max_active = 0

    async def fake_run_canary(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        artifact_dir = kwargs["artifact_dir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        patch = artifact_dir / "patch.diff"
        prompt = artifact_dir / "prompt.txt"
        command = artifact_dir / "command.json"
        workspace = artifact_dir / "workspace_snapshot"
        grade_ref = artifact_dir / "grade.json"
        patch.write_text("diff --git a/x b/x\n")
        prompt.write_text("Fix it")
        command.write_text("{}")
        workspace.mkdir()
        grade_ref.write_text("{}")
        return RolloutRecord(
            rollout_id=kwargs["rollout_id"],
            task_id="generated_repo_tasks__x",
            source_name="generated_repo_tasks",
            capability="agentic_coding",
            harness="opencode",
            workflow=Workflow(steps=[WorkflowStep(worker_id=2, subtask="fix")]),
            execution=Execution(
                steps=[
                    ExecStep(
                        worker_id=2,
                        harness="opencode",
                        session_ref=str(workspace),
                        patch_ref=str(patch),
                        messages_ref=str(prompt),
                        tool_events_ref=str(command),
                        text=patch.read_text(),
                    )
                ]
            ),
            grade=Grade(score=1.0, success=True, grader_ref=str(grade_ref)),
            reward=1.0,
        )

    monkeypatch.setattr("ultra.trace_capture_run.run_canary", fake_run_canary)

    report = await run_trace_capture_jobs(
        jobs_jsonl=jobs_jsonl,
        report_out=tmp_path / "report.json",
        parallel=2,
    )

    assert report["counts"] == {"ok": 2}
    assert report["parallel"] == 2
    assert max_active == 2
