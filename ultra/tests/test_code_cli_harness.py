import subprocess
from pathlib import Path

import pytest

from ultra.config import WorkerSpec
from ultra.harness import HARNESS_REGISTRY, StepInput, StepResult
from ultra.harness.code_cli import (
    ClaudeCodeHarness,
    CodexHarness,
    YunwuAnthropicProxy,
    _anthropic_messages_to_openai,
    _openai_message_to_anthropic,
)
from ultra.schemas import EnvironmentSpec, GraderSpec, SourceRef, SplittingSpec, TaskInput, TaskSpec
from ultra.workers import FakeProvider, Sampling, WorkerPool


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="repo-task",
        capability="agentic_coding",
        source=SourceRef(name="s", version="v", policy="train_allowed"),
        input=TaskInput(
            messages=[{"role": "user", "content": "Fix the bug"}],
            assets=[
                {
                    "opencode_instance": {
                        "image_name": "example/task:latest",
                        "instance_id": "buggy",
                        "problem_statement": "Fix the bug",
                        "testbed": "/testbed",
                    }
                }
            ],
        ),
        environment=EnvironmentSpec(harness="codex", wall_time_seconds=120),
        grader=GraderSpec(type="deep_swe_hidden_tests"),
        splitting=SplittingSpec(group_id="g", split="pool_validation"),
    )


def _pool() -> WorkerPool:
    return WorkerPool([WorkerSpec(worker_id="codex_gpt_coding_agent", model="gpt-5.5")], FakeProvider())


def test_codex_and_claude_harnesses_registered_as_concrete_adapters():
    assert HARNESS_REGISTRY["codex"] is CodexHarness
    assert HARNESS_REGISTRY["claude_code"] is ClaudeCodeHarness


def test_cli_commands_are_noninteractive_and_workspace_scoped(monkeypatch):
    monkeypatch.setenv("ULTRA_ALLOW_YUNWU", "1")
    monkeypatch.delenv("ULTRA_CODEX_PROVIDER", raising=False)
    monkeypatch.delenv("ULTRA_CODEX_WIRE_API", raising=False)
    codex = CodexHarness()._command("/bin/codex", "gpt-5-codex", Path("/repo"))
    assert codex[:2] == ["/bin/codex", "exec"]
    assert "--cd" in codex
    assert "/repo" in codex
    assert "model_provider=\"yunwu\"" in codex
    assert "model_providers.yunwu.base_url=\"https://yunwu.ai/v1\"" in codex
    assert "model_providers.yunwu.env_key=\"YUNWU_API_KEY\"" in codex
    assert "--dangerously-bypass-approvals-and-sandbox" in codex

    claude = ClaudeCodeHarness()._command("/bin/claude", "claude-opus-4.8", Path("/repo"))
    assert claude[:2] == ["/bin/claude", "-p"]
    assert "--add-dir" in claude
    assert "/repo" in claude
    assert "--safe-mode" in claude
    assert "--no-session-persistence" in claude


def test_cli_model_overrides_and_aliases(monkeypatch):
    monkeypatch.delenv("ULTRA_PROVIDER", raising=False)
    assert ClaudeCodeHarness()._cli_model("claude-opus-4.8") == "opus"
    assert ClaudeCodeHarness()._upstream_model() == "claude-opus-4-8"
    monkeypatch.setenv("ULTRA_CLAUDE_CLI_MODEL", "sonnet")
    monkeypatch.setenv("ULTRA_CLAUDE_MODEL", "claude-custom")
    assert ClaudeCodeHarness()._cli_model("claude-opus-4.8") == "sonnet"
    assert ClaudeCodeHarness()._upstream_model() == "claude-custom"
    monkeypatch.delenv("ULTRA_CODEX_MODEL", raising=False)
    assert CodexHarness()._cli_model("gpt-5-codex") == "gpt-5.5"
    monkeypatch.setenv("ULTRA_CODEX_MODEL", "gpt-custom")
    assert CodexHarness()._cli_model("gpt-5-codex") == "gpt-custom"


def test_codex_provider_overrides_can_select_wire_api(monkeypatch):
    monkeypatch.setenv("ULTRA_ALLOW_YUNWU", "1")
    monkeypatch.setenv("ULTRA_CODEX_WIRE_API", "responses")

    command = CodexHarness()._command("/bin/codex", "gpt-5.5", Path("/repo"))

    assert "model_providers.yunwu.wire_api=\"responses\"" in command


def test_anthropic_messages_translate_to_openai_chat():
    payload = {
        "system": "You are a coding agent.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Fix it"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.py"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "value = 1"}]},
        ],
    }

    messages = _anthropic_messages_to_openai(payload)

    assert messages[0] == {"role": "system", "content": "You are a coding agent."}
    assert messages[1] == {"role": "user", "content": "Fix it"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "Read"
    assert messages[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "value = 1"}


def test_openai_chat_response_translates_to_anthropic_tool_use():
    message = _openai_message_to_anthropic(
        {
            "id": "chatcmpl-1",
            "model": "claude-opus-4-8",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Edit", "arguments": "{\"file_path\":\"a.py\"}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
        "claude-opus-4-8",
    )

    assert message["stop_reason"] == "tool_use"
    assert message["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "Edit",
        "input": {"file_path": "a.py"},
    }


def test_claude_harness_uses_local_yunwu_anthropic_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("ULTRA_ALLOW_YUNWU", "1")

    class FakeProxy:
        def __init__(self, *, provider_name, model):
            assert provider_name == "yunwu"
            assert model == "claude-opus-4-8"
            self.stopped = False

        def start(self):
            return "http://127.0.0.1:4567"

        def stop(self):
            self.stopped = True

    captured = {}

    class FakePopen:
        pid = 12345

        def __init__(self, args, *, cwd, stdin, stdout, stderr, text, env, start_new_session):
            captured["args"] = args
            captured["env"] = env
            self.returncode = 0

        def communicate(self, input=None, timeout=None):
            return "done", ""

    monkeypatch.delenv("ULTRA_CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("ULTRA_CLAUDE_PROVIDER", raising=False)
    monkeypatch.setattr("ultra.harness.code_cli.YunwuAnthropicProxy", FakeProxy)
    monkeypatch.setattr("ultra.harness.code_cli.subprocess.Popen", FakePopen)
    monkeypatch.setattr("ultra.harness.code_cli.os.killpg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("ultra.harness.code_cli.time.sleep", lambda _seconds: None)

    result = ClaudeCodeHarness()._run_cli(
        "/bin/claude",
        "opus",
        tmp_path,
        "Fix it",
        10,
    )

    assert result["status"] == "ok"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4567"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "ultra-local-yunwu-proxy"
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in captured["env"]


def test_codex_harness_blocks_yunwu_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ULTRA_ALLOW_YUNWU", raising=False)

    with pytest.raises(RuntimeError, match="Yunwu Codex call"):
        CodexHarness()._command("/bin/codex", "gpt-5.5", Path("/repo"))


def test_claude_harness_blocks_yunwu_without_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("ULTRA_ALLOW_YUNWU", raising=False)

    with pytest.raises(RuntimeError, match="Yunwu Claude Code call"):
        ClaudeCodeHarness()._run_cli("/bin/claude", "opus", tmp_path, "Fix it", 10)


def test_yunwu_proxy_accepts_messages_query_paths(monkeypatch):
    monkeypatch.setenv("YUNWU_API_KEY", "test")
    proxy = YunwuAnthropicProxy(provider_name="yunwu", model="claude-opus-4-8")
    base_url = proxy.start()
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base_url}/v1/messages/count_tokens?anthropic-beta=tools",
            data=b'{"messages":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = response.read().decode()
        assert "input_tokens" in payload
    finally:
        proxy.stop()


@pytest.mark.asyncio
async def test_codex_harness_fails_closed_when_cli_missing(monkeypatch):
    monkeypatch.setattr("ultra.harness.code_cli._cli_binary", lambda env, default: None)

    result = await CodexHarness().run_step(
        StepInput(task=_task(), subtask="Implement the fix", worker_id="codex_gpt_coding_agent"),
        _pool(),
        Sampling(),
    )

    assert result.termination == "missing_cli"
    assert "Codex CLI not found" in result.error


@pytest.mark.asyncio
async def test_codex_harness_runs_cli_on_exported_workspace_and_grades(monkeypatch, tmp_path):
    class FakeContainer:
        def __init__(self, image, instance_id, *, testbed="/testbed", tests_dir=None):
            self.image = image
            self.instance_id = instance_id
            self.testbed = testbed
            self.tests_dir = tests_dir

        def start(self):
            return True

        def export_workspace(self, destination):
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=destination, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "ultra@example.invalid"],
                cwd=destination,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Ultra"],
                cwd=destination,
                check=True,
                capture_output=True,
            )
            (destination / "bug.py").write_text("value = 1\n")
            subprocess.run(["git", "add", "bug.py"], cwd=destination, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=destination, check=True, capture_output=True)
            return True, ""

        def grade_deep_swe(self, diff):
            return 1.0 if "value = 2" in diff else 0.0

        def close(self):
            return None

    def fake_run_cli(self, binary, model, workspace, prompt, timeout):
        assert binary == "/fake/codex"
        assert model == "gpt-5.5"
        assert "Fix the bug" in prompt
        (workspace / "bug.py").write_text("value = 2\n")
        return {"status": "ok", "returncode": 0, "stdout": "done", "stderr": ""}

    monkeypatch.setenv("ULTRA_CODECLI_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setattr("ultra.harness.code_cli._cli_binary", lambda env, default: "/fake/codex")
    monkeypatch.setattr("ultra.harness.code_cli.RepoTaskContainer", FakeContainer)
    monkeypatch.setattr(CodexHarness, "_run_cli", fake_run_cli)

    harness = CodexHarness()
    result = await harness.run_step(
        StepInput(
            task=_task(),
            subtask="Implement the fix",
            worker_id="codex_gpt_coding_agent",
            artifact_dir=str(tmp_path / "artifacts" / "step0"),
        ),
        _pool(),
        Sampling(),
    )
    grade = harness.grade(_task(), result)

    assert result.termination == "completed"
    assert "value = 2" in result.text
    assert result.patch_ref and Path(result.patch_ref).exists()
    assert result.messages_ref and Path(result.messages_ref).exists()
    assert result.tool_events_ref and Path(result.tool_events_ref).exists()
    assert result.workspace_snapshot_ref and Path(result.workspace_snapshot_ref).exists()
    assert grade.success is True


@pytest.mark.asyncio
async def test_codex_harness_applies_initial_patch_before_cli(monkeypatch, tmp_path):
    initial_patch = tmp_path / "initial.patch"
    initial_patch.write_text(
        "\n".join(
            [
                "diff --git a/bug.py b/bug.py",
                "--- a/bug.py",
                "+++ b/bug.py",
                "@@ -1 +1 @@",
                "-value = 1",
                "+value = 2",
                "",
            ]
        )
    )

    task = _task()
    task.input.assets[0]["opencode_instance"]["initial_patch_ref"] = str(initial_patch)

    class FakeContainer:
        def __init__(self, image, instance_id, *, testbed="/testbed", tests_dir=None):
            self.image = image
            self.instance_id = instance_id
            self.testbed = testbed
            self.tests_dir = tests_dir

        def start(self):
            return True

        def export_workspace(self, destination):
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=destination, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "ultra@example.invalid"],
                cwd=destination,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Ultra"],
                cwd=destination,
                check=True,
                capture_output=True,
            )
            (destination / "bug.py").write_text("value = 1\n")
            subprocess.run(["git", "add", "bug.py"], cwd=destination, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=destination, check=True, capture_output=True)
            return True, ""

        def close(self):
            return None

    def fake_run_cli(self, binary, model, workspace, prompt, timeout):
        assert (workspace / "bug.py").read_text() == "value = 2\n"
        assert "previous engineer's partial attempt" in prompt
        (workspace / "bug.py").write_text("value = 3\n")
        return {"status": "ok", "returncode": 0, "stdout": "done", "stderr": ""}

    monkeypatch.setenv("ULTRA_CODECLI_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setattr("ultra.harness.code_cli._cli_binary", lambda env, default: "/fake/codex")
    monkeypatch.setattr("ultra.harness.code_cli.RepoTaskContainer", FakeContainer)
    monkeypatch.setattr(CodexHarness, "_run_cli", fake_run_cli)

    result = await CodexHarness().run_step(
        StepInput(task=task, subtask="Repair the previous patch", worker_id="codex_gpt_coding_agent"),
        _pool(),
        Sampling(),
    )

    assert result.termination == "completed"
    assert "value = 3" in result.text


def test_code_cli_harness_grades_nonzero_patch():
    class FakeContainer:
        def grade_deep_swe(self, diff):
            return 1.0 if "patch" in diff else 0.0

    task = _task()
    harness = CodexHarness()
    harness.instance = task.input.assets[0]["opencode_instance"]
    harness.final_container = FakeContainer()

    grade = harness.grade(
        task,
        StepResult(text="patch", error="Codex exited 1", termination="cli_nonzero_with_patch"),
    )

    assert grade.success is True
    assert grade.details["step_error"] == "Codex exited 1"
