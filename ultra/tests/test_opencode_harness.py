import pytest

from ultra.config import WorkerSpec
from ultra.harness import HARNESS_REGISTRY, StepInput
from ultra.harness.opencode import KEY_ENV, OpenCodeRepoHarness, _deep_swe_reward_from_text, _opencode_cost
from ultra.schemas import EnvironmentSpec, GraderSpec, SourceRef, SplittingSpec, TaskInput, TaskSpec
from ultra.workers import FakeProvider, Sampling, WorkerPool


def test_opencode_cost_sums_json_step_finish_events():
    stdout = "\n".join(
        [
            '{"type":"text","part":{"text":"ok"}}',
            '{"type":"step_finish","part":{"cost":0.25}}',
            "not json",
            '{"type":"step_finish","part":{"cost":0.125}}',
        ]
    )
    assert _opencode_cost(stdout) == 0.375


def test_deep_swe_reward_parser_accepts_json_and_text_sentinels():
    assert _deep_swe_reward_from_text('{"reward": 1, "f2p": 1.0}') == 1.0
    assert _deep_swe_reward_from_text('{"reward": 0, "f2p": 0.0}') == 0.0
    assert _deep_swe_reward_from_text("1") == 1.0
    assert _deep_swe_reward_from_text("-1") == 0.0
    assert _deep_swe_reward_from_text("not-json") == 0.0


def test_opencode_harness_registered_under_backend_and_legacy_names():
    assert HARNESS_REGISTRY["opencode"] is OpenCodeRepoHarness
    assert "opencode_repo" in HARNESS_REGISTRY


@pytest.mark.asyncio
async def test_opencode_harness_fails_closed_without_repo_payload(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "test-key")
    task = TaskSpec(
        task_id="repo-task",
        capability="agentic_coding",
        source=SourceRef(name="s", version="v", policy="train_allowed"),
        input=TaskInput(messages=[{"role": "user", "content": "Fix the bug"}]),
        environment=EnvironmentSpec(harness="opencode_repo"),
        grader=GraderSpec(type="hidden_tests"),
        splitting=SplittingSpec(group_id="g", split="pool_validation"),
    )
    pool = WorkerPool([WorkerSpec(worker_id="opencode_kimi_builder", model="fake/model")], FakeProvider())
    result = await OpenCodeRepoHarness().run_step(
        StepInput(task=task, subtask="Implement the fix", worker_id="opencode_kimi_builder"),
        pool,
        Sampling(),
    )
    assert result.termination == "missing_task_payload"
    assert "opencode_instance" in result.error
