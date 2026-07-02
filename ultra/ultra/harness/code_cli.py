"""Codex and Claude Code repository harnesses.

These adapters execute coding-assistant CLIs against a host-side copy of the
repository workspace, then grade the produced patch through the same hidden-test
payload used by the OpenCode repo harness. That keeps Codex/Claude auth and local
configuration outside task containers while preserving the benchmark patch/grade
contract.
"""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
import urllib.error
from urllib.parse import urlparse
import urllib.request

from ..providers import assert_live_provider_allowed
from ..providers import provider as _provider_cfg
from ..providers import routed_provider_name
from ..providers import slug as _provider_slug
from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness, wall_time_cap_seconds
from .opencode import (
    TESTBED,
    _deep_swe_reward_from_text,
    _normalize_instance,
    _read_initial_patch,
    _step_prompt,
    _strip_ignored_diff_entries,
)
from .repo_artifacts import artifact_ref, copy_workspace, write_json, write_repo_state, write_text

CODECLI_WORKDIR_ENV = "ULTRA_CODECLI_WORKDIR"
CODECLI_KEEP_WORKDIR_ENV = "ULTRA_CODECLI_KEEP_WORKDIR"
CODECLI_TIMEOUT = int(os.environ.get("ULTRA_CODECLI_TIMEOUT", "1800"))
CLAUDE_API_TIMEOUT_MS = os.environ.get("ULTRA_CLAUDE_API_TIMEOUT_MS", "3000000")


def _sh(*args: str, cwd: Path | None = None, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, input=input_text, capture_output=True, text=True, check=False)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:120] or "task"


def _cli_binary(env_name: str, default: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else None
    return shutil.which(default)


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _workspace_root() -> Path:
    return Path(os.environ.get(CODECLI_WORKDIR_ENV, ".ultra_codecli_runs")).resolve()


def _git_diff(workspace: Path) -> tuple[bool, str, str]:
    add = _sh("git", "add", "-A", cwd=workspace)
    if add.returncode != 0:
        return False, "", (add.stderr or add.stdout).strip()
    diff = _sh("git", "diff", "--cached", cwd=workspace)
    if diff.returncode != 0:
        return False, "", (diff.stderr or diff.stdout).strip()
    return True, _strip_ignored_diff_entries(diff.stdout), ""


def _apply_initial_patch_to_workspace(workspace: Path, patch: str) -> tuple[bool, str]:
    proc = _sh("git", "apply", "--whitespace=nowarn", "-", cwd=workspace, input_text=patch)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, ""


def _anthropic_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text") or ""))
        elif block_type == "tool_result":
            result = block.get("content")
            parts.append(_anthropic_text(result))
    return "\n".join(part for part in parts if part)


def _anthropic_content_to_openai_messages(role: str, content: Any) -> list[dict[str, Any]]:
    if role == "assistant" and isinstance(content, list):
        text = "\n".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        tool_calls = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"toolu_{len(tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return [message]

    if role == "user" and isinstance(content, list):
        messages: list[dict[str, Any]] = []
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        text = "\n".join(str(b.get("text") or "") for b in text_blocks)
        if text:
            messages.append({"role": "user", "content": text})
        for block in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or block.get("id") or ""),
                    "content": _anthropic_text(block.get("content")),
                }
            )
        return messages or [{"role": "user", "content": _anthropic_text(content)}]

    return [{"role": role, "content": _anthropic_text(content)}]


def _anthropic_messages_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_text(system)})
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        messages.extend(_anthropic_content_to_openai_messages(role, message.get("content")))
    return messages


def _anthropic_tools_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return None


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _openai_finish_to_anthropic(reason: str | None) -> str:
    if reason == "tool_calls":
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "end_turn"


def _openai_message_to_anthropic(upstream: dict[str, Any], fallback_model: str) -> dict[str, Any]:
    choice = (upstream.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if content:
        content_blocks.append({"type": "text", "text": content if isinstance(content, str) else json.dumps(content)})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = {"_raw_arguments": raw_args}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"toolu_{len(content_blocks)}"),
                "name": str(function.get("name") or ""),
                "input": parsed_args,
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    usage = upstream.get("usage") or {}
    return {
        "id": str(upstream.get("id") or f"msg_{int(time.time() * 1000)}"),
        "type": "message",
        "role": "assistant",
        "model": str(upstream.get("model") or fallback_model),
        "content": content_blocks,
        "stop_reason": _openai_finish_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _sse(handler: BaseHTTPRequestHandler, event: str, payload: dict[str, Any]) -> None:
    handler.wfile.write(f"event: {event}\n".encode())
    handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())


def _stream_anthropic_response(handler: BaseHTTPRequestHandler, message: dict[str, Any]) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    started = {k: v for k, v in message.items() if k != "content"}
    started["content"] = []
    started["stop_reason"] = None
    _sse(handler, "message_start", {"type": "message_start", "message": started})
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            _sse(handler, "content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
            if block.get("text"):
                _sse(handler, "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}})
        else:
            _sse(
                handler,
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}},
                },
            )
            _sse(
                handler,
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})},
                },
            )
        _sse(handler, "content_block_stop", {"type": "content_block_stop", "index": index})
    _sse(
        handler,
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        },
    )
    _sse(handler, "message_stop", {"type": "message_stop"})


class YunwuAnthropicProxy:
    """Local Claude Messages facade over Yunwu's OpenAI-compatible chat API."""

    def __init__(self, *, provider_name: str, model: str) -> None:
        cfg = _provider_cfg(provider_name)
        self.provider_name = provider_name
        self.model = model
        self.base_url = str(cfg["base_url"]).rstrip("/")
        self.key_env = str(cfg.get("key_env") or "")
        self.api_key = os.environ.get(self.key_env) if self.key_env else None
        self.provider_sort = os.environ.get("ULTRA_CLAUDE_PROVIDER_SORT", "price")
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - stdlib hook
                path = urlparse(self.path).path.rstrip("/")
                if path in {"", "/v1/models", "/models"}:
                    _json_response(self, 200, {"data": [{"id": proxy.model, "type": "model"}]})
                    return
                _json_response(self, 404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802 - stdlib hook
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except Exception:
                    _json_response(self, 400, {"error": {"message": "invalid json"}})
                    return
                path = urlparse(self.path).path.rstrip("/")
                if path.endswith("/messages/count_tokens"):
                    approx = max(1, len(json.dumps(payload)) // 4)
                    _json_response(self, 200, {"input_tokens": approx})
                    return
                if not path.endswith("/messages"):
                    _json_response(self, 404, {"error": {"message": "not found"}})
                    return
                if not proxy.api_key:
                    _json_response(self, 401, {"error": {"message": f"missing {proxy.key_env}"}})
                    return
                proxy.handle_messages(self, payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)

    def handle_messages(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        upstream_body: dict[str, Any] = {
            "model": self.model,
            "messages": _anthropic_messages_to_openai(payload),
            "stream": False,
            "provider": {"sort": self.provider_sort},
        }
        for src, dst in [
            ("max_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ]:
            if payload.get(src) is not None:
                upstream_body[dst] = payload[src]
        tools = _anthropic_tools_to_openai(payload)
        if tools:
            upstream_body["tools"] = tools
        tool_choice = _anthropic_tool_choice_to_openai(payload.get("tool_choice"))
        if tool_choice:
            upstream_body["tool_choice"] = tool_choice
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(upstream_body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(30, CODECLI_TIMEOUT)) as response:
                upstream = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-1000:]
            _json_response(handler, 502, {"error": {"message": f"yunwu upstream HTTP {exc.code}: {detail}"}})
            return
        except Exception as exc:
            _json_response(handler, 502, {"error": {"message": f"yunwu upstream error: {type(exc).__name__}: {exc}"}})
            return
        message = _openai_message_to_anthropic(upstream, str(upstream_body["model"]))
        if payload.get("stream"):
            _stream_anthropic_response(handler, message)
        else:
            _json_response(handler, 200, message)


class RepoTaskContainer:
    """Container for branch checkout and hidden-test grading, not for the CLI loop."""

    def __init__(
        self,
        image: str,
        instance_id: str,
        *,
        testbed: str = TESTBED,
        tests_dir: str | None = None,
    ) -> None:
        self.image = image
        self.instance_id = instance_id
        self.testbed = testbed
        self.tests_dir = tests_dir
        self.cid = ""

    def start(self) -> bool:
        run_args = ["docker", "run", "-d", "--rm"]
        if self.tests_dir:
            run_args += ["-v", f"{self.tests_dir}:/tests:ro"]
        run_args += [self.image, "sleep", "9000"]
        proc = _sh(*run_args)
        cid = proc.stdout.strip()
        if not cid:
            return False
        if self.instance_id:
            co = _sh("docker", "exec", cid, "bash", "-lc", f"cd {self.testbed} && git checkout {self.instance_id} 2>&1")
            blob = (co.stderr + co.stdout).lower()
            if "error" in blob and "set up to track" not in blob:
                _sh("docker", "rm", "-f", cid)
                return False
        self.cid = cid
        return True

    def export_workspace(self, destination: Path) -> tuple[bool, str]:
        destination.mkdir(parents=True, exist_ok=True)
        proc = _sh("docker", "cp", f"{self.cid}:{self.testbed}/.", str(destination))
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()
        return True, ""

    def grade_deep_swe(self, diff: str, artifact_dir: Path | None = None) -> float:
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.cid,
                "bash",
                "-lc",
                "mkdir -p /logs/artifacts /logs/verifier && cat > /logs/artifacts/model.patch",
            ],
            input=diff,
            capture_output=True,
            text=True,
            check=False,
        )
        proc = subprocess.run(
            ["docker", "exec", self.cid, "bash", "-lc", "bash /tests/test.sh"],
            capture_output=True,
            text=True,
            timeout=CODECLI_TIMEOUT,
            check=False,
        )
        reward = _sh(
            "docker",
            "exec",
            self.cid,
            "bash",
            "-lc",
            "cat /logs/verifier/reward.json 2>/dev/null || cat /logs/verifier/reward.txt 2>/dev/null || true",
        ).stdout
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            write_text(
                artifact_dir / "test_command.log",
                f"returncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}",
            )
            write_text(artifact_dir / "reward_raw.txt", reward)
        return _deep_swe_reward_from_text(reward)

    def close(self) -> None:
        if self.cid:
            subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)


class _CodeCliRepoHarness:
    name: str
    cli_label: str
    bin_env: str
    default_bin: str

    def __init__(self) -> None:
        self.containers: dict[int, RepoTaskContainer] = {}
        self.workspaces: dict[int, Path] = {}
        self.diffs: dict[int, str] = {}
        self.owned_containers: list[RepoTaskContainer] = []
        self.owned_workdirs: list[Path] = []
        self.instance: dict[str, Any] | None = None
        self.final_container: RepoTaskContainer | None = None
        self.step_artifact_dirs: dict[int, Path] = {}

    def _command(self, binary: str, model: str, workspace: Path) -> list[str]:
        raise NotImplementedError

    def _cli_model(self, model: str) -> str:
        return model

    def _run_cli(
        self,
        binary: str,
        model: str,
        workspace: Path,
        prompt: str,
        timeout: int | None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                self._command(binary, model, workspace),
                cwd=workspace,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate(prompt, timeout=timeout)
            _terminate_process_group(proc.pid)
            return {
                "status": "ok" if proc.returncode == 0 else "nonzero",
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        except subprocess.TimeoutExpired as exc:
            if proc is not None:
                _terminate_process_group(proc.pid)
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
            else:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
            return {
                "status": "timeout",
                "returncode": None,
                "stdout": stdout or "",
                "stderr": stderr or "",
            }

    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult:
        del sampling

        instance = _normalize_instance(step.task)
        if instance is None:
            return StepResult(
                text="",
                error=f"{self.name} task is missing an opencode_instance payload",
                termination="missing_task_payload",
            )
        self.instance = instance

        binary = _cli_binary(self.bin_env, self.default_bin)
        if binary is None:
            return StepResult(
                text="",
                error=f"{self.cli_label} CLI not found; set {self.bin_env} or install {self.default_bin}",
                termination="missing_cli",
            )

        access = list(step.access)
        if len(access) == 1 and access[0] in self.workspaces and access[0] in self.containers:
            workspace = self.workspaces[access[0]]
            container = self.containers[access[0]]
            prior_diffs: list[str] = []
            continuing = True
        else:
            container = RepoTaskContainer(
                str(instance["image_name"]),
                str(instance.get("instance_id", "")),
                testbed=str(instance.get("testbed") or TESTBED),
                tests_dir=str(instance["tests_dir"]) if instance.get("tests_dir") else None,
            )
            if not await asyncio.to_thread(container.start):
                return StepResult(
                    text="",
                    error="container start/checkout failed",
                    termination="container_start_failed",
                )
            self.owned_containers.append(container)

            run_dir = (
                _workspace_root()
                / f"{_safe_slug(step.task.task_id)}__{self.name}__s{step.step_index}__{int(time.time())}"
            )
            workspace = run_dir / "repo"
            exported, export_error = await asyncio.to_thread(container.export_workspace, workspace)
            if not exported:
                return StepResult(text="", error=export_error, termination="workspace_export_failed")
            self.owned_workdirs.append(run_dir)

            initial_patch, initial_patch_error = _read_initial_patch(instance)
            if initial_patch_error:
                return StepResult(
                    text="",
                    error=initial_patch_error,
                    termination="initial_patch_failed",
                )
            if initial_patch:
                applied, apply_error = await asyncio.to_thread(
                    _apply_initial_patch_to_workspace,
                    workspace,
                    initial_patch,
                )
                if not applied:
                    return StepResult(
                        text="",
                        error=apply_error,
                        termination="initial_patch_failed",
                    )

            artifacts = {a.get("step_index"): a for a in step.prior_artifacts}
            prior_diffs = []
            for j in access:
                if j in self.diffs:
                    prior_diffs.append(self.diffs[j])
                elif j in artifacts:
                    prior_diffs.append(str(artifacts[j].get("response", "")))
            continuing = bool(prior_diffs or instance.get("initial_patch_ref"))

        prompt = _step_prompt(
            str(instance["problem_statement"]),
            step.subtask,
            prior_diffs,
            continuing,
            testbed=str(workspace),
        )
        timeout = wall_time_cap_seconds(
            step.budget,
            task_cap=step.task.environment.wall_time_seconds,
            harness_cap=CODECLI_TIMEOUT,
        )
        model = self._cli_model(pool.model_for(step.worker_id))
        run = await asyncio.to_thread(self._run_cli, binary, model, workspace, prompt, timeout)

        diff_ok, diff, diff_error = await asyncio.to_thread(_git_diff, workspace)
        if not diff_ok:
            return StepResult(text="", error=diff_error, termination="diff_failed")

        self.diffs[step.step_index] = diff
        self.workspaces[step.step_index] = workspace
        self.containers[step.step_index] = container
        self.final_container = container
        refs: dict[str, str | None] = {
            "messages_ref": None,
            "patch_ref": None,
            "tool_events_ref": None,
            "workspace_snapshot_ref": None,
        }
        step_artifact_dir = Path(step.artifact_dir) if step.artifact_dir else None
        if step_artifact_dir is not None:
            self.step_artifact_dirs[step.step_index] = step_artifact_dir
            refs["messages_ref"] = write_text(step_artifact_dir / "prompt.txt", prompt)
            if instance.get("initial_patch_ref"):
                write_json(
                    step_artifact_dir / "initial_patch.json",
                    {"initial_patch_ref": str(instance.get("initial_patch_ref"))},
                )
            refs["patch_ref"] = write_text(step_artifact_dir / "patch.diff", diff)
            refs["tool_events_ref"] = write_json(
                step_artifact_dir / "command.json",
                {
                    "harness": self.name,
                    "worker_id": step.worker_id,
                    "status": run.get("status"),
                    "returncode": run.get("returncode"),
                    "stdout": run.get("stdout", ""),
                    "stderr": run.get("stderr", ""),
                    "timeout_seconds": timeout,
                    "workspace": str(workspace),
                },
            )
            write_repo_state(step_artifact_dir / "repo_state.json", step.task, instance)
            refs["workspace_snapshot_ref"] = copy_workspace(
                workspace,
                step_artifact_dir / "workspace_snapshot",
            )

        status = str(run["status"])
        if status == "timeout":
            return StepResult(
                text=diff,
                error=f"{self.cli_label} timed out",
                termination="timeout",
                session_ref=refs["workspace_snapshot_ref"],
                workspace_snapshot_ref=refs["workspace_snapshot_ref"],
                patch_ref=refs["patch_ref"],
                messages_ref=refs["messages_ref"],
                tool_events_ref=refs["tool_events_ref"],
                command_log_ref=refs["tool_events_ref"],
                artifact_dir=str(step_artifact_dir) if step_artifact_dir is not None else None,
            )
        if status != "ok" and not diff.strip():
            stderr = str(run.get("stderr") or run.get("stdout") or "").strip()
            message = f"{self.cli_label} exited {run.get('returncode')}"
            if stderr:
                message = f"{message}: {stderr[-1000:]}"
            return StepResult(
                text=diff,
                error=message,
                termination="cli_failed",
                session_ref=refs["workspace_snapshot_ref"],
                workspace_snapshot_ref=refs["workspace_snapshot_ref"],
                patch_ref=refs["patch_ref"],
                messages_ref=refs["messages_ref"],
                tool_events_ref=refs["tool_events_ref"],
                command_log_ref=refs["tool_events_ref"],
                artifact_dir=str(step_artifact_dir) if step_artifact_dir is not None else None,
            )
        return StepResult(
            text=diff,
            error=None,
            termination="completed" if status == "ok" else "cli_nonzero_with_patch",
            session_ref=refs["workspace_snapshot_ref"],
            workspace_snapshot_ref=refs["workspace_snapshot_ref"],
            patch_ref=refs["patch_ref"],
            messages_ref=refs["messages_ref"],
            tool_events_ref=refs["tool_events_ref"],
            command_log_ref=refs["tool_events_ref"],
            artifact_dir=str(step_artifact_dir) if step_artifact_dir is not None else None,
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        try:
            if final.error and not final.text.strip():
                return Grade(score=0.0, success=False, details={"error": final.error})
            patch = _strip_ignored_diff_entries(final.text)
            if not patch.strip():
                return Grade(score=0.0, success=False, details={"error": "empty patch"})
            instance = self.instance or _normalize_instance(task)
            if not instance:
                return Grade(
                    score=0.0,
                    success=False,
                    details={"error": f"{self.name} task is missing an opencode_instance payload"},
                )
            if task.grader.type == "deep_swe_hidden_tests":
                if self.final_container is None:
                    return Grade(
                        score=0.0,
                        success=False,
                        details={"error": "deep_swe_hidden_tests has no final container"},
                    )
                grade_dir = Path(final.artifact_dir) / "grade" if final.artifact_dir else None
                try:
                    reward = float(self.final_container.grade_deep_swe(patch, grade_dir))
                except TypeError:
                    reward = float(self.final_container.grade_deep_swe(patch))
                details = {"step_error": final.error} if final.error else {}
                if grade_dir is not None:
                    details.update(
                        {
                            "public_test_log_ref": artifact_ref(grade_dir / "test_command.log"),
                            "hidden_grade_ref": artifact_ref(grade_dir / "reward_raw.txt"),
                        }
                    )
                return Grade(score=reward, success=reward >= task.grader.success_threshold, details=details)
            if task.grader.type == "swebench_verified_hidden_tests":
                from ..acrouter_swebench import grade_swebench_verified_patch

                instance_id = str(instance.get("swebench_instance_id") or instance.get("instance_id") or "")
                if not instance_id:
                    return Grade(
                        score=0.0,
                        success=False,
                        details={"error": "missing swebench_instance_id"},
                    )
                grade_dir = (
                    Path(final.artifact_dir) / "grade"
                    if final.artifact_dir
                    else Path(".ultra_swebench_grades") / task.task_id
                )
                result = grade_swebench_verified_patch(
                    instance_id=instance_id,
                    patch=patch,
                    image=str(instance["image_name"]),
                    log_dir=grade_dir,
                    eval_timeout=task.environment.wall_time_seconds or CODECLI_TIMEOUT,
                    network="none",
                )
                reward = 1.0 if result.get("resolved") else 0.0
                details = {"step_error": final.error} if final.error else {}
                details.update(
                    {
                        "swebench_instance_id": instance_id,
                        "apply_ok": result.get("apply_ok"),
                        "resolved": result.get("resolved"),
                        "error": result.get("error"),
                        "redacted_log_path": result.get("redacted_log_path"),
                        "raw_log_retained": result.get("raw_log_retained"),
                    }
                )
                return Grade(score=reward, success=reward >= task.grader.success_threshold, details=details)
            if task.grader.type in {"hidden_tests", "swesmith_hidden_tests"}:
                from director.agentic.swebench_mini import grade_swesmith

                reward = float(grade_swesmith(instance, patch))
                details = {"step_error": final.error} if final.error else {}
                return Grade(score=reward, success=reward >= task.grader.success_threshold, details=details)
            return Grade(
                score=0.0,
                success=False,
                details={"error": f"unsupported {self.name} grader {task.grader.type!r}"},
            )
        except Exception as exc:  # noqa: BLE001 - live graders fail heterogeneously
            return Grade(score=0.0, success=False, details={"error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.close()

    def close(self) -> None:
        seen_containers: set[int] = set()
        for container in self.owned_containers:
            ident = id(container)
            if ident not in seen_containers:
                seen_containers.add(ident)
                container.close()
        if os.environ.get(CODECLI_KEEP_WORKDIR_ENV) == "1":
            return
        seen_workdirs: set[Path] = set()
        for workdir in self.owned_workdirs:
            if workdir not in seen_workdirs:
                seen_workdirs.add(workdir)
                shutil.rmtree(workdir, ignore_errors=True)


@register_harness
class CodexHarness(_CodeCliRepoHarness):
    name = "codex"
    cli_label = "Codex"
    bin_env = "ULTRA_CODEX_BIN"
    default_bin = "codex"

    def _cli_model(self, model: str) -> str:
        return os.environ.get("ULTRA_CODEX_MODEL") or _provider_slug("gpt")

    def _provider_config_args(self) -> list[str]:
        provider_name = routed_provider_name("gpt", os.environ.get("ULTRA_CODEX_PROVIDER"))
        assert_live_provider_allowed(provider_name, model="gpt", context="Codex call")
        cfg = _provider_cfg(provider_name)
        args = [
            "-c",
            f"model_provider={_toml_string(provider_name)}",
            "-c",
            f"model_providers.{provider_name}.name={_toml_string(provider_name)}",
            "-c",
            f"model_providers.{provider_name}.base_url={_toml_string(str(cfg['base_url']))}",
        ]
        key_env = str(cfg.get("key_env") or "")
        if key_env:
            args.extend(["-c", f"model_providers.{provider_name}.env_key={_toml_string(key_env)}"])
        wire_api = os.environ.get("ULTRA_CODEX_WIRE_API")
        if wire_api:
            args.extend(["-c", f"model_providers.{provider_name}.wire_api={_toml_string(wire_api)}"])
        return args

    def _command(self, binary: str, model: str, workspace: Path) -> list[str]:
        return [
            binary,
            "exec",
            "--model",
            model,
            *self._provider_config_args(),
            "--cd",
            str(workspace),
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--dangerously-bypass-approvals-and-sandbox",
            "-",
        ]


@register_harness
class ClaudeCodeHarness(_CodeCliRepoHarness):
    name = "claude_code"
    cli_label = "Claude Code"
    bin_env = "ULTRA_CLAUDE_BIN"
    default_bin = "claude"

    def _cli_model(self, model: str) -> str:
        return os.environ.get("ULTRA_CLAUDE_CLI_MODEL", "opus")

    def _upstream_model(self) -> str:
        return os.environ.get("ULTRA_CLAUDE_MODEL") or _provider_slug("opus")

    def _run_cli(
        self,
        binary: str,
        model: str,
        workspace: Path,
        prompt: str,
        timeout: int | None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        provider_name = routed_provider_name("opus", os.environ.get("ULTRA_CLAUDE_PROVIDER"))
        assert_live_provider_allowed(provider_name, model="opus", context="Claude Code call")
        upstream_model = self._upstream_model()
        proxy = YunwuAnthropicProxy(provider_name=provider_name, model=upstream_model)
        base_url = proxy.start()
        run_env = dict(os.environ if env is None else env)
        run_env.pop("ANTHROPIC_API_KEY", None)
        run_env.pop("ANTHROPIC_DEFAULT_OPUS_MODEL", None)
        run_env.pop("ANTHROPIC_DEFAULT_SONNET_MODEL", None)
        run_env.pop("ANTHROPIC_DEFAULT_HAIKU_MODEL", None)
        run_env.update(
            {
                "ANTHROPIC_AUTH_TOKEN": os.environ.get("ANTHROPIC_AUTH_TOKEN", "ultra-local-yunwu-proxy"),
                "ANTHROPIC_BASE_URL": base_url,
                "API_TIMEOUT_MS": CLAUDE_API_TIMEOUT_MS,
            }
        )
        try:
            return super()._run_cli(binary, model, workspace, prompt, timeout, env=run_env)
        finally:
            proxy.stop()

    def _command(self, binary: str, model: str, workspace: Path) -> list[str]:
        return [
            binary,
            "-p",
            "--model",
            model,
            "--safe-mode",
            "--permission-mode",
            "bypassPermissions",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--input-format",
            "text",
            "--output-format",
            "text",
            "--add-dir",
            str(workspace),
        ]
