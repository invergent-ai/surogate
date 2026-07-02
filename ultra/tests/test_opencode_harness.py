import pytest

from ultra.config import WorkerSpec
from ultra.harness import HARNESS_REGISTRY, StepInput, StepResult
from ultra.harness.opencode import (
    KEY_ENV,
    OpenCodeRepoHarness,
    _deep_swe_reward_from_text,
    _opencode_cost,
    _opencode_route,
    _strip_ignored_diff_entries,
)
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


def test_diff_sanitizer_drops_generated_bytecode_blocks():
    diff = "\n".join(
        [
            "diff --git a/__pycache__/slugkit.cpython-311.pyc b/__pycache__/slugkit.cpython-311.pyc",
            "new file mode 100644",
            "Binary files /dev/null and b/__pycache__/slugkit.cpython-311.pyc differ",
            "diff --git a/slugkit.py b/slugkit.py",
            "--- a/slugkit.py",
            "+++ b/slugkit.py",
            "@@ -1 +1 @@",
            "-bad",
            "+good",
            "",
        ]
    )
    clean = _strip_ignored_diff_entries(diff)
    assert "__pycache__" not in clean
    assert "slugkit.py" in clean


def test_opencode_harness_registered_under_backend_and_legacy_names():
    assert HARNESS_REGISTRY["opencode"] is OpenCodeRepoHarness
    assert "opencode_repo" in HARNESS_REGISTRY


def test_opencode_open_workers_default_to_openrouter():
    route = _opencode_route("opencode_kimi_builder")

    assert route["provider_name"] == "openrouter"
    assert route["oc_provider"] == "openrouter"
    assert route["key_env"] == "OPENROUTER_API_KEY"
    assert route["oc_config"] == ""
    assert route["slug"] == "moonshotai/kimi-k2.7-code"


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


def test_opencode_harness_grades_timeout_patch():
    class FakeContainer:
        def grade_deep_swe(self, diff):
            return 1.0 if diff.strip() else 0.0

    task = TaskSpec(
        task_id="repo-task",
        capability="agentic_coding",
        source=SourceRef(name="s", version="v", policy="train_allowed"),
        input=TaskInput(
            messages=[{"role": "user", "content": "Fix the bug"}],
            assets=[
                {
                    "opencode_instance": {
                        "image_name": "example/task:latest",
                        "problem_statement": "Fix the bug",
                    }
                }
            ],
        ),
        environment=EnvironmentSpec(harness="opencode"),
        grader=GraderSpec(type="deep_swe_hidden_tests"),
        splitting=SplittingSpec(group_id="g", split="pool_validation"),
    )
    harness = OpenCodeRepoHarness()
    harness.instance = task.input.assets[0]["opencode_instance"]
    harness.final_container = FakeContainer()

    grade = harness.grade(task, StepResult(text="diff --git a/x b/x\n", error="timeout", termination="timeout"))

    assert grade.success is True
    assert grade.details["step_error"] == "timeout"
