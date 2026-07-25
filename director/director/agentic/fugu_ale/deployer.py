from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, ClassVar
from urllib.request import urlopen

from ale_run.agents.terminus_2.deployer import Terminus2Deployer
from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig, TaskOS
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from ultra.pool_binding import load_pool_binding
from ultra.tool_actions import (
    COMPUTER_ACTION_WORKER_CONTRACT,
    ToolActionContractError,
    parse_computer_action,
)

from director.agentic.fugu_ultra_terminal import (
    PRODUCT_POOL_BINDING,
    PRODUCT_RUNTIME_REVISION,
    ALLOWED_WORKER_PROVIDER_BASES,
    WORKER_PROVIDER_KEY_ENV,
    YUNWU_API_BASE,
    FuguUltraTerminalAgent,
)

from .config import FuguAleConfig
from .cua import AleCuaClient, AleCuaError, AleCuaObservation

ALE_FUGU_RUNTIME_REVISION = "20260722-ale-linux-cli-cua-v10-fail-closed"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALE_DOUBLE_FORK = """\
import os
import sys

start_log = sys.argv[1]
command = sys.argv[2]
if os.fork():
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
os.chdir("/")
stdin_fd = os.open("/dev/null", os.O_RDONLY)
log_fd = os.open(start_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.dup2(stdin_fd, 0)
os.dup2(log_fd, 1)
os.dup2(log_fd, 2)
os.closerange(3, 1024)
os.execl("/bin/bash", "bash", "-lc", command)
"""


class AleRemoteEnvironment(BaseEnvironment):
    """Harbor environment facade over ALE's host-side sandbox handle."""

    def __init__(self, executor: Any, *, trial_dir: Path, task_budget_s: float) -> None:
        trial_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.sandbox = executor.sandbox
        self.environment_dir = trial_dir
        self.environment_name = "ale-remote"
        self.session_id = f"ale-{self.sandbox.id}"
        self.trial_paths = TrialPaths(trial_dir=trial_dir)
        self.trial_paths.mkdir()
        self.default_user = None
        self.task_env_config = EnvironmentConfig(os=TaskOS.LINUX)
        self.agent_timeout_s = float(task_budget_s)
        self._persistent_env: dict[str, str] = {}

    @staticmethod
    def type() -> str:
        return "ale-remote"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=False,
            disable_internet=False,
            mounted=False,
            windows=False,
        )

    @property
    def task_os(self) -> TaskOS:
        return TaskOS.LINUX

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool = False) -> None:
        del force_build

    async def stop(self, delete: bool = False) -> None:
        del delete

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        shell_command = command
        if env:
            invalid = sorted(name for name in env if not _ENV_NAME.fullmatch(name))
            if invalid:
                raise ValueError(f"invalid environment variable names: {invalid}")
            assignments = " ".join(
                f"{name}={shlex.quote(str(value))}" for name, value in env.items()
            )
            shell_command = f"env {assignments} {shell_command}"
        if cwd:
            shell_command = f"cd {shlex.quote(cwd)} && {shell_command}"
        if user is not None:
            shell_command = (
                f"sudo -n -u {shlex.quote(str(user))} "
                f"bash -lc {shlex.quote(shell_command)}"
            )
        else:
            shell_command = f"bash -lc {shlex.quote(shell_command)}"

        result = await self.sandbox.run_command(
            shell_command,
            timeout=float(timeout_sec or 60),
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.sandbox.upload_local_file(os.fspath(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        for path in source.rglob("*"):
            relative = path.relative_to(source).as_posix()
            remote = f"{target_dir.rstrip('/')}/{relative}"
            if path.is_dir():
                await self.sandbox.mkdir(remote)
            else:
                await self.sandbox.upload_local_file(os.fspath(path), remote)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = await self.sandbox.download_to_local(
            source_path,
            os.fspath(target),
            timeout=300,
        )
        if not copied:
            raise FileNotFoundError(source_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        for entry in await self.sandbox.list_dir(source_dir):
            relative = Path(entry["relpath"])
            destination = target / relative
            if entry["is_dir"]:
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            remote = f"{source_dir.rstrip('/')}/{relative.as_posix()}"
            copied = await self.sandbox.download_to_local(
                remote,
                os.fspath(destination),
                timeout=300,
            )
            if not copied:
                raise FileNotFoundError(remote)


class AleTmuxSession(TmuxSession):
    """Tmux transport that cannot strand the ALE command response stream.

    The stock combined ``new-session ; pipe-pane`` command does not return
    through cua-server: its pane pipe remains live until the session exits.
    Harbor already captures every observation with ``capture-pane`` and writes
    the canonical trajectory host-side, so the ALE transport deliberately
    omits that redundant streaming pane log and starts only the detached shell.
    """

    @property
    def _tmux_start_session(self) -> str:
        env_options = "".join(
            f"-e {shlex.quote(f'{key}={value}')} "
            for key, value in self._extra_env.items()
        )
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session {env_options}"
            f"-x {self._pane_width} -y {self._pane_height} "
            f"-d -s {shlex.quote(self._session_name)} 'bash --login' "
            "</dev/null >/dev/null 2>&1"
        )

    def _detached_start_command(self, start_log: str) -> str:
        return " ".join(
            (
                "python3",
                "-c",
                shlex.quote(_ALE_DOUBLE_FORK),
                shlex.quote(start_log),
                shlex.quote(self._tmux_start_session),
            )
        )

    async def start(self) -> None:
        await self._attempt_tmux_installation()
        start_log = f"/tmp/{self._session_name}.start.log"
        launch_result = await self.environment.exec(
            command=self._detached_start_command(start_log),
            user=self._user,
        )
        if launch_result.return_code != 0:
            raise RuntimeError(
                "failed to launch detached ALE tmux session: "
                f"{launch_result.stderr.strip()}"
            )
        await asyncio.sleep(0.2)
        if not await self.is_session_alive():
            start_result = await self.environment.exec(
                command=f"cat {shlex.quote(start_log)} 2>/dev/null || true",
                user=self._user,
            )
            raise RuntimeError(
                "detached tmux launch returned without a live ALE session: "
                f"{start_result.stdout.strip()}"
            )

        history_result = await self.environment.exec(
            command="tmux set-option -g history-limit 10000000",
            user=self._user,
        )
        if history_result.return_code != 0:
            raise RuntimeError(
                "failed to set ALE tmux history limit: "
                f"{history_result.stderr.strip()}"
            )

        if self._remote_asciinema_recording_path:
            await self.send_keys(
                keys=[
                    f"asciinema rec --stdin {self._remote_asciinema_recording_path}",
                    "Enter",
                ],
                min_timeout_sec=1.0,
            )
            await self.send_keys(keys=["clear", "Enter"])
            await self.environment.upload_file(
                source_path=self._GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH,
                target_path=str(self.GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH),
            )


class FuguUltraAleAgent(FuguUltraTerminalAgent):
    """Fugu's Linux terminal loop without TerminalBench workspace policing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logs_dir = kwargs.get("logs_dir") or (args[0] if args else None)
        if logs_dir is None:
            raise TypeError("logs_dir is required")
        binding_path = Path(kwargs.get("pool_binding_path", PRODUCT_POOL_BINDING))
        binding = load_pool_binding(binding_path)
        kwargs.setdefault(
            "worker_tool_tags",
            {
                slot.worker_id: (
                    "terminal",
                    "filesystem",
                    "test_runner",
                    "computer_use",
                )
                for slot in binding.slots
            },
        )
        kwargs.setdefault("worker_tool_contract", COMPUTER_ACTION_WORKER_CONTRACT)
        super().__init__(*args, **kwargs)
        self._ale_logs_dir = Path(logs_dir)
        self._cua_client: AleCuaClient | None = None
        self._cua_events: list[dict[str, Any]] = []
        self._cua_events_path = self._ale_logs_dir / "cua_events.jsonl"

    @staticmethod
    def name() -> str:
        return "fugu-ultra-ale"

    @staticmethod
    def tmux_session_name(environment: BaseEnvironment) -> str:
        identity = str(
            getattr(environment, "session_id", "")
            or getattr(getattr(environment, "sandbox", None), "id", "")
            or "unknown"
        )
        suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return f"fugu-ultra-ale-{suffix}"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Create one isolated, finite-start tmux session per ALE sandbox."""
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = AleTmuxSession(
            session_name=self.tmux_session_name(environment),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    def version(self) -> str | None:
        return f"{PRODUCT_RUNTIME_REVISION}-{ALE_FUGU_RUNTIME_REVISION}"

    async def _prepare_workspace_snapshot(self, environment: BaseEnvironment) -> None:
        del environment

    async def _ensure_workspace_integrity(self) -> bool:
        return False

    async def _remove_workspace_sentinel(self) -> None:
        return None

    async def _execute_commands(
        self,
        commands: list[Any],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Fail closed as soon as ALE's persistent shell disappears."""
        for index, command in enumerate(commands, start=1):
            if not await session.is_session_alive():
                raise RuntimeError(
                    "ALE tmux session terminated before command "
                    f"{index}/{len(commands)}; refusing to recover or replay commands"
                )
            try:
                await session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
            except TimeoutError:
                return True, self._timeout_template.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        await session.get_incremental_output()
                    ),
                )
            if not await session.is_session_alive():
                raise RuntimeError(
                    "ALE tmux session terminated after command "
                    f"{index}/{len(commands)}; refusing to recover or replay commands"
                )

        if not await session.is_session_alive():
            raise RuntimeError(
                "ALE tmux session terminated before terminal capture; "
                "refusing to recover or replay commands"
            )
        return False, self._limit_output_length(await session.get_incremental_output())

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        sandbox = getattr(environment, "sandbox", None)
        endpoint = getattr(sandbox, "endpoint", None)
        if not isinstance(endpoint, str) or not endpoint:
            raise RuntimeError("ALE computer actions require a sandbox endpoint")
        self._cua_client = AleCuaClient(endpoint)
        self._cua_events.clear()
        self._cua_events_path.unlink(missing_ok=True)
        try:
            await super().run(instruction, environment, context)
        finally:
            metadata = dict(context.metadata or {})
            metadata.update(
                {
                    "cua_action_count": len(self._cua_events),
                    "cua_error_count": sum(
                        bool(event.get("is_error")) for event in self._cua_events
                    ),
                    "cua_actions": [
                        {
                            key: value
                            for key, value in event.items()
                            if key != "image_base64"
                        }
                        for event in self._cua_events
                    ],
                }
            )
            context.metadata = metadata
            self._cua_client = None

    def _record_cua_event(
        self,
        action_name: str,
        arguments: dict[str, Any],
        observation: AleCuaObservation,
    ) -> None:
        screenshot_path: str | None = None
        if observation.image_base64 is not None:
            screenshot_dir = self._ale_logs_dir / "cua_screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot = screenshot_dir / f"{len(self._cua_events):04d}.png"
            screenshot.write_bytes(base64.b64decode(observation.image_base64))
            screenshot_path = str(screenshot)
        event = {
            "event_index": len(self._cua_events),
            "runtime_turn": self._fugu_llm.runtime_turns,
            "action": action_name,
            "arguments": arguments,
            "text": observation.text,
            "is_error": observation.is_error,
            "screenshot_path": screenshot_path,
            "media_type": observation.media_type,
        }
        self._cua_events.append(event)
        self._cua_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cua_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    def _return_cua_observation(
        self,
        action_name: str,
        arguments: dict[str, Any],
        observation: AleCuaObservation,
    ) -> str:
        self._fugu_llm.record_active_tool_observation(
            tool_name=action_name,
            text=observation.text,
            is_error=observation.is_error,
            image_base64=observation.image_base64,
            media_type=observation.media_type or "image/png",
        )
        self._record_cua_event(action_name, arguments, observation)
        status = "error" if observation.is_error else "result"
        return f"WARNINGS: computer.{action_name} {status}: {observation.text}"

    async def _handle_llm_interaction(
        self,
        chat: Any,
        prompt: str,
        logging_paths: tuple[Any | None, Any | None, Any | None] = (
            None,
            None,
            None,
        ),
        original_instruction: str = "",
        session: Any = None,
    ) -> tuple[list[Any], bool, str, str, str, Any]:
        interaction = await super()._handle_llm_interaction(
            chat,
            prompt,
            logging_paths,
            original_instruction,
            session,
        )
        commands, is_task_complete, feedback, analysis, plan, response = interaction
        try:
            payload = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return interaction
        if not isinstance(payload, dict) or "computer_action" not in payload:
            return interaction
        try:
            action = parse_computer_action(payload)
        except ToolActionContractError as exc:
            observation = AleCuaObservation(
                action="invalid",
                text=str(exc),
                is_error=True,
            )
            tool_feedback = self._return_cua_observation("invalid", {}, observation)
            return [], False, tool_feedback, analysis, plan, response
        if action is None:
            return interaction
        if commands:
            observation = AleCuaObservation(
                action=action.name,
                text="terminal commands and computer_action are mutually exclusive",
                is_error=True,
            )
            tool_feedback = self._return_cua_observation(
                action.name,
                action.arguments,
                observation,
            )
            return [], False, tool_feedback, analysis, plan, response
        if is_task_complete:
            observation = AleCuaObservation(
                action=action.name,
                text="task_complete cannot accompany a computer action",
                is_error=True,
            )
            tool_feedback = self._return_cua_observation(
                action.name,
                action.arguments,
                observation,
            )
            return [], False, tool_feedback, analysis, plan, response
        if self._cua_client is None:
            raise RuntimeError("ALE CUA client is not initialized")
        try:
            observation = await self._cua_client.execute(action)
        except (AleCuaError, OSError, ValueError) as exc:
            observation = AleCuaObservation(
                action=action.name,
                text=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        tool_feedback = self._return_cua_observation(
            action.name,
            action.arguments,
            observation,
        )
        if feedback:
            tool_feedback = f"{feedback}\n{tool_feedback}"
        return [], False, tool_feedback, analysis, plan, response


class FuguAleDeployer(Terminus2Deployer):
    """Run Fugu host-side while its terminal session lives in an ALE sandbox."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "logs/agent/trajectory.json",
        "fugu_context.json",
    )

    @property
    def version(self) -> str | None:
        return ALE_FUGU_RUNTIME_REVISION

    async def install(self) -> None:
        cfg: FuguAleConfig = self.config
        if not self.executor.sandbox.is_linux:
            raise NotImplementedError(
                "Fugu ALE v1 supports only the Linux CLI task surface"
            )
        provider_base = cfg.provider_base_url.rstrip("/")
        if provider_base not in ALLOWED_WORKER_PROVIDER_BASES:
            raise ValueError(
                f"paid workers must use one of {ALLOWED_WORKER_PROVIDER_BASES}"
            )
        provider_key_env = WORKER_PROVIDER_KEY_ENV[provider_base]
        if not os.environ.get(provider_key_env):
            raise RuntimeError(
                f"{provider_key_env} is required for paid workers on {provider_base}"
            )
        if cfg.record_conductor_token_data and not cfg.fail_closed_provider_errors:
            raise RuntimeError(
                "training collection must abort on the first provider failure"
            )
        if cfg.record_conductor_token_data and (
            isinstance(cfg.conductor_temperature, bool)
            or not isinstance(cfg.conductor_temperature, (int, float))
            or float(cfg.conductor_temperature) != 1.0
        ):
            raise RuntimeError(
                "training conductor temperature must be exactly 1.0"
            )

        models_url = f"{cfg.conductor_base_url.rstrip('/')}/models"

        def fetch_models() -> dict[str, Any]:
            with urlopen(models_url, timeout=10) as response:  # noqa: S310 - local server
                return json.load(response)

        payload = await asyncio.to_thread(fetch_models)
        served_rows = [
            row for row in payload.get("data", []) if isinstance(row, dict)
        ]
        served = {row.get("id") for row in served_rows}
        if cfg.model not in served:
            raise RuntimeError(
                f"conductor {cfg.model!r} is not served at {models_url!r}"
            )
        if cfg.record_conductor_token_data:
            revision = str(cfg.conductor_policy_revision or "").strip()
            if not revision:
                raise RuntimeError(
                    "training collection requires conductor_policy_revision"
                )
            model_row = next(row for row in served_rows if row.get("id") == cfg.model)
            served_revisions = {str(model_row.get("id"))}
            for field in ("root", "parent"):
                value = model_row.get(field)
                if isinstance(value, str) and value:
                    served_revisions.add(value)
                    served_revisions.add(Path(value).name)
            adapter_root = model_row.get("root")
            if isinstance(adapter_root, str) and Path(adapter_root).is_dir():
                policy_manifest = Path(adapter_root) / "fugu_policy_revision.json"
                if policy_manifest.is_file():
                    try:
                        registered_revision = json.loads(
                            policy_manifest.read_text(encoding="utf-8")
                        ).get("policy_revision")
                    except (OSError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            "served conductor policy-revision manifest is invalid"
                        ) from exc
                    if isinstance(registered_revision, str) and registered_revision:
                        served_revisions.add(registered_revision)
            if revision not in served_revisions:
                raise RuntimeError(
                    "served conductor does not attest the registered behavior-policy "
                    f"revision {revision!r}"
                )
            if (
                cfg.optimizer_sequence_len != 2_816
                or cfg.conductor_max_input_tokens <= 0
                or cfg.conductor_max_output_tokens <= 0
                or cfg.conductor_max_input_tokens + cfg.conductor_max_output_tokens
                > cfg.optimizer_sequence_len
            ):
                raise RuntimeError(
                    "ALE training collection must fit the 8,192-token "
                    "optimizer window"
                )

        work_dir = Path(self.executor.work_dir)
        (work_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)
        (work_dir / "harbor_trial").mkdir(parents=True, exist_ok=True)

    def _build_agent(self, cfg: FuguAleConfig, logs_dir: Path) -> FuguUltraAleAgent:
        kwargs: dict[str, Any] = {
            "logs_dir": logs_dir,
            "max_turns": cfg.max_turns,
            "provider_base_url": cfg.provider_base_url,
            "fail_closed_provider_errors": cfg.fail_closed_provider_errors,
            "typed_conductor_model": (
                None if cfg.solo_worker_id is not None else cfg.model
            ),
            "typed_conductor_url": cfg.conductor_base_url,
            "typed_conductor_temperature": cfg.conductor_temperature,
            "typed_conductor_seed": cfg.conductor_seed,
            "typed_conductor_record_token_data": cfg.record_conductor_token_data,
            "typed_conductor_policy_revision": cfg.conductor_policy_revision,
            "typed_conductor_max_input_tokens": cfg.conductor_max_input_tokens,
            "typed_conductor_max_output_tokens": cfg.conductor_max_output_tokens,
            "record_terminal_session": cfg.record_terminal_session,
            "worker_session_namespace": f"fugu-ale-{self.executor.sandbox.id}",
        }
        if cfg.pool_binding_path:
            kwargs["pool_binding_path"] = cfg.pool_binding_path
        if cfg.solo_worker_id is not None:
            kwargs["solo_worker_id"] = cfg.solo_worker_id
        return FuguUltraAleAgent(**kwargs)

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: FuguAleConfig = self.config
        work_dir = Path(self.executor.work_dir)
        logs_dir = work_dir / "logs" / "agent"
        context_path = work_dir / "fugu_context.json"
        stderr_path = work_dir / "stderr.log"
        environment = AleRemoteEnvironment(
            self.executor,
            trial_dir=work_dir / "harbor_trial",
            task_budget_s=cfg.task_budget_s,
        )
        agent = self._build_agent(cfg, logs_dir)
        session_name = FuguUltraAleAgent.tmux_session_name(environment)
        quoted_session_name = shlex.quote(session_name)
        context = AgentContext()
        started = time.monotonic()
        stderr_path.write_text("", encoding="utf-8")

        prepare_result = await environment.exec(
            "install -d -m 0777 /logs/agent",
            timeout_sec=30,
            user="root",
        )
        if prepare_result.return_code != 0:
            raise RuntimeError(
                "failed to prepare writable ALE agent log directory: "
                f"{prepare_result.stderr.strip()}"
            )
        await environment.exec(
            f"tmux kill-session -t {quoted_session_name} 2>/dev/null || true",
            timeout_sec=30,
        )
        try:
            await agent.setup(environment)
            await agent.run(prompt, environment, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - return an auditable failed run
            stderr_path.write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            return AgentRunResult(
                status="failed",
                transcript_path=str(logs_dir / "trajectory.json"),
                stderr_path=str(stderr_path),
                duration_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            session = getattr(agent, "_session", None)
            if session is not None:
                try:
                    await session.stop()
                except Exception:  # noqa: BLE001 - cleanup must preserve run status
                    pass
            try:
                await environment.exec(
                    f"tmux kill-session -t {quoted_session_name} 2>/dev/null || true",
                    timeout_sec=30,
                )
            except Exception:  # noqa: BLE001 - outer ALE lifecycle owns the sandbox
                pass
            context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")

        return AgentRunResult(
            status="completed",
            transcript_path=str(logs_dir / "trajectory.json"),
            stderr_path=str(stderr_path),
            duration_s=time.monotonic() - started,
        )

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: FuguAleConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        super().parse_artifacts(
            work_dir=work_dir,
            config=config,
            run_result=run_result,
            builder=builder,
        )
        context_path = work_dir / "fugu_context.json"
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        builder.trajectory.extra["fugu"] = {
            "runtime_revision": ALE_FUGU_RUNTIME_REVISION,
            "metadata": context.get("metadata") or {},
        }
