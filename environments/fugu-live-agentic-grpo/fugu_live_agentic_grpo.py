"""On-policy GRPO environment for the product Fugu live conductor.

Each rollout runs the real Harbor agent and functional verifier. The trainable
model supplies either compact live decisions or, in the candidate protocol,
the complete initial/replacement topology and live actions. Workers, tools,
isolated trajectories, shared memory, and grading stay inside the product runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import verifiers as vf
from datasets import Dataset
from verifiers.utils.message_utils import normalize_messages


def _ensure_repo_imports() -> Path:
    root = Path(__file__).resolve().parents[2]
    for path in (root, root / "ultra", root / "director"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


ROOT = _ensure_repo_imports()

from ultra.live_control import (  # noqa: E402
    LIVE_AGENTIC_GRPO_BRIDGE_VERSION,
    MAX_DECISION_CORRECTIONS,
    ControlBudget,
    ControlContractError,
    ControlPosition,
    LiveControlState,
    WorkerProfile,
    build_control_action_messages,
    build_control_decision_messages,
    capability_reference_map,
    canonicalize_control_decision,
    parse_control_action,
    parse_capability_control_action,
    parse_control_decision,
    render_control_action_correction,
    render_decision_correction,
    validate_control_action,
    validate_control_decision,
)


BRIDGE_VERSION = LIVE_AGENTIC_GRPO_BRIDGE_VERSION
POOL_BINDING_PATH = (
    ROOT
    / "director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json"
)
_POOL_BINDING = json.loads(POOL_BINDING_PATH.read_text(encoding="utf-8"))
DEFAULT_WORKER_MODELS = tuple(
    slot["runtime_model"] for slot in _POOL_BINDING["slots"]
)
YUNWU_API_BASE = str(_POOL_BINDING["provider_base"])
POOL_FINGERPRINT = str(_POOL_BINDING["pool_fingerprint"])
MAX_AGENT_TURNS = 120
MAX_CONTROL_DECISIONS = MAX_AGENT_TURNS + 2
PRODUCT_RUNTIME_REVISION = "20260720-r58-capability-set-interface"


ENV_VERSION = "20260720-live-agentic-grpo-env-v6-capability-contract"
DEFAULT_MANIFEST = ROOT / "scratchpad/fugu_live_agentic_grpo_pilot/tasks.jsonl"
DEFAULT_ARTIFACT_ROOT = ROOT / "scratchpad/fugu_live_agentic_grpo_pilot/rollouts"
HARBOR = ROOT / "director/.venv/bin/harbor"
AGENT_IMPORT_PATH = "director.agentic.fugu_live_grpo:FuguLiveGRPOAgent"
LOCAL_AGENT_IMPORT_PATH = (
    "director.agentic.fugu_mechanics_grpo:FuguMechanicsGRPOAgent"
)
LOCAL_MECHANICS_BINDING = (
    ROOT / "director/manifests/fugu_mechanics/mechanics_pool_binding_v1_composite.json"
)
_LOCAL_BINDING_RAW = json.loads(LOCAL_MECHANICS_BINDING.read_text(encoding="utf-8"))
LOCAL_WORKER_MODELS = tuple(
    slot["runtime_model"] for slot in _LOCAL_BINDING_RAW["slots"]
)
LOCAL_PROVIDER_BASE = str(_LOCAL_BINDING_RAW["provider_base"])
LOCAL_CONTROLLER_ADAPTER = "mechanics-composite-v1"
POLL_INTERVAL_S = 0.1
PROTECTED_TEST_POLICY = "prepared_index_test_blobs_restored_after_each_batch"


def _control_protocol(*, unified_control: bool, capability_refs: bool) -> str:
    if capability_refs:
        if not unified_control:
            raise ValueError("capability references require unified topology control")
        return "unified_capability_action_v2"
    return "unified_full_action_v1" if unified_control else "compact_live_decision_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        content = (
            last.get("content")
            if isinstance(last, dict)
            else getattr(last, "content", None)
        )
        if isinstance(content, list):
            content = "".join(
                str(
                    part.get("text", "")
                    if isinstance(part, dict)
                    else getattr(part, "text", "")
                )
                for part in content
            )
        return str(content or "")
    return str(completion or "")


def _info(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("rollout info must be an object")
    return value


def _numeric_reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"live-agentic GRPO manifest is missing: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("live-agentic GRPO manifest is empty")
    task_ids: set[str] = set()
    for row in rows:
        required = {
            "task_id",
            "task_dir",
            "task_tree_sha256",
            "oracle_result_path",
            "oracle_result_sha256",
            "oracle_reward",
            "source_policy",
            "conductor_attempted_before",
        }
        if set(row) != required:
            raise ValueError(
                f"manifest row has unexpected schema for {row.get('task_id')!r}"
            )
        task_id = str(row["task_id"])
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id in manifest: {task_id}")
        task_ids.add(task_id)
        task_dir = Path(row["task_dir"]).expanduser().resolve()
        oracle = Path(row["oracle_result_path"]).expanduser().resolve()
        if row["source_policy"] != "train_allowed":
            raise ValueError(f"task is not train-allowed: {task_id}")
        if row["conductor_attempted_before"] is not False:
            raise ValueError(f"task was already attempted by a conductor: {task_id}")
        if not task_dir.is_dir() or _tree_sha256(task_dir) != row["task_tree_sha256"]:
            raise ValueError(f"task tree changed after registration: {task_id}")
        if not oracle.is_file() or _sha256(oracle) != row["oracle_result_sha256"]:
            raise ValueError(f"oracle evidence changed after registration: {task_id}")
        if row["oracle_reward"] != 1.0 or _numeric_reward(
            json.loads(oracle.read_text(encoding="utf-8"))
        ) != 1.0:
            raise ValueError(f"task lacks an audited functional oracle pass: {task_id}")
    return rows


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    request: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    returncode: int | None = None
    error: str | None = None


class ControlSession(Protocol):
    async def start(self) -> None: ...

    async def next_event(self, after_request_id: int) -> SessionEvent: ...

    async def submit(self, request_id: int, completion: str) -> None: ...

    async def close(self) -> None: ...


class HarborControlSession:
    """One Harbor task attempt connected to the external controller bridge."""

    def __init__(
        self,
        *,
        rollout_id: str,
        task_dir: Path,
        artifact_root: Path,
        event_timeout_s: float,
        worker_timeout_s: float,
        decision_timeout_s: float,
        provider_mode: str = "live",
        unified_control: bool = False,
        capability_refs: bool = False,
    ) -> None:
        self.provider_mode = provider_mode
        self.rollout_id = rollout_id
        self.task_dir = task_dir
        self.root = artifact_root / rollout_id
        self.control_dir = self.root / "control"
        self.run_dir = self.root / "harbor"
        self.log_path = self.root / "harbor.log"
        self.event_timeout_s = event_timeout_s
        self.worker_timeout_s = worker_timeout_s
        self.decision_timeout_s = decision_timeout_s
        self.unified_control = bool(unified_control)
        self.capability_refs = bool(capability_refs)
        _control_protocol(
            unified_control=self.unified_control,
            capability_refs=self.capability_refs,
        )
        self.process: asyncio.subprocess.Process | None = None
        self._log_handle: Any | None = None

    async def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("Harbor session already started")
        if not HARBOR.is_file():
            raise RuntimeError(f"Harbor executable is missing: {HARBOR}")
        if self.provider_mode == "live" and not os.environ.get("YUNWU_API_KEY"):
            raise RuntimeError("YUNWU_API_KEY is required for live GRPO workers")
        self.control_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": f"{ROOT / 'director'}:{ROOT / 'ultra'}:{ROOT}",
                "FUGU_TB_TOTAL_S": str(self.worker_timeout_s),
                "FUGU_TB_FIRST_CONTENT_S": str(self.worker_timeout_s),
                "FUGU_GRPO_CONTROL_DIR": str(self.control_dir),
                "FUGU_GRPO_ROLLOUT_ID": self.rollout_id,
                "FUGU_GRPO_DECISION_TIMEOUT_S": str(self.decision_timeout_s),
                "FUGU_GRPO_UNIFIED_CONTROL": "1" if self.unified_control else "0",
                "FUGU_GRPO_CAPABILITY_REFS": "1" if self.capability_refs else "0",
            }
        )
        if self.provider_mode == "local":
            env.pop("YUNWU_API_KEY", None)
            env.update(
                {
                    "FUGU_MECHANICS_LIVE_BINDING": str(LOCAL_MECHANICS_BINDING),
                    "FUGU_MECHANICS_LIVE_CONTROLLER": LOCAL_CONTROLLER_ADAPTER,
                    "FUGU_MECHANICS_LIVE_COLLECTION_ID": self.rollout_id,
                }
            )
        else:
            env["ULTRA_ALLOW_YUNWU"] = "1"
        agent_import = (
            LOCAL_AGENT_IMPORT_PATH
            if self.provider_mode == "local"
            else AGENT_IMPORT_PATH
        )
        command = [
            str(HARBOR),
            "run",
            "-p",
            str(self.task_dir),
            "--agent-import-path",
            agent_import,
            "-m",
            f"fugu-live-agentic-grpo/{'local-mechanics' if self.provider_mode == 'local' else 'current-pool'}",
            "-l",
            "1",
            "-n",
            "1",
            "-o",
            str(self.run_dir),
            "--job-name",
            "rollout",
            "-y",
        ]
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            env=env,
            stdout=self._log_handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

    def _request(self, request_id: int) -> dict[str, Any] | None:
        path = self.control_dir / f"request_{request_id:04d}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != BRIDGE_VERSION:
            raise RuntimeError("controller request bridge version mismatch")
        if payload.get("rollout_id") != self.rollout_id:
            raise RuntimeError("controller request rollout ID mismatch")
        if payload.get("request_id") != request_id:
            raise RuntimeError("controller request ID mismatch")
        expected_protocol = _control_protocol(
            unified_control=self.unified_control,
            capability_refs=self.capability_refs,
        )
        if payload.get("control_protocol") != expected_protocol:
            raise RuntimeError("controller request protocol mismatch")
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or [message.get("role") for message in messages] != ["system", "user"]
        ):
            raise RuntimeError("controller request messages are invalid")
        return payload

    def _result(self) -> dict[str, Any] | None:
        paths = sorted(self.run_dir.glob("rollout/*/result.json"))
        if not paths:
            paths = sorted(self.run_dir.glob("**/result.json"))
        if len(paths) > 1:
            raise RuntimeError(f"Harbor produced {len(paths)} result files")
        if not paths:
            return None
        return json.loads(paths[0].read_text(encoding="utf-8"))

    async def next_event(self, after_request_id: int) -> SessionEvent:
        if self.process is None:
            raise RuntimeError("Harbor session is not started")
        wanted = after_request_id + 1
        deadline = time.monotonic() + self.event_timeout_s
        while True:
            request = self._request(wanted)
            if request is not None:
                return SessionEvent(kind="request", request=request)
            if self.process.returncode is not None:
                result = self._result()
                if result is None:
                    return SessionEvent(
                        kind="error",
                        returncode=self.process.returncode,
                        error="Harbor exited without a result",
                    )
                return SessionEvent(
                    kind="result",
                    result=result,
                    returncode=self.process.returncode,
                )
            if time.monotonic() >= deadline:
                return SessionEvent(
                    kind="error",
                    error=f"no controller boundary or result within {self.event_timeout_s:g}s",
                )
            await asyncio.sleep(POLL_INTERVAL_S)

    async def submit(self, request_id: int, completion: str) -> None:
        response = self.control_dir / f"response_{request_id:04d}.json"
        if response.exists():
            raise RuntimeError(f"controller request {request_id} already answered")
        temporary = response.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": BRIDGE_VERSION,
                    "rollout_id": self.rollout_id,
                    "request_id": request_id,
                    "completion": completion,
                    "created_at_unix_s": time.time(),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, response)

    async def close(self) -> None:
        process = self.process
        if process is not None and process.returncode is None:
            cancel = self.control_dir / "cancel.json"
            cancel.write_text('{"cancelled":true}\n', encoding="utf-8")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _fake_state(
    *,
    completion_requested: bool,
    initial: bool = False,
    worker_id: int = 2,
    subtask: str = "Build and verify the concrete repair.",
) -> LiveControlState:
    return LiveControlState(
        original_task="Repair the supplied repository and verify the result.",
        workers=tuple(
            WorkerProfile(
                worker_id=index,
                capability_tags=(f"task_specific_role_{index}",),
                tool_tags=("terminal", "filesystem", "test_runner"),
            )
            for index in range(4)
        ),
        workflow_id=None if initial else 1,
        positions=() if initial else (
            ControlPosition(
                position_id=0,
                worker_id=worker_id,
                subtask=subtask,
                access=(),
                status="active",
                progress={
                    "completion_requested": completion_requested,
                    "worker_report": "verified" if completion_requested else "editing",
                    "turns": 2 if completion_requested else 1,
                },
                artifacts=({"path": "/app/module.py", "state": "modified"},),
            ),
        ),
        active_position_id=None if initial else 0,
        terminal_status="ready",
        terminal_observation="root@task:/app# pytest -q\n1 passed\nroot@task:/app#",
        shared_memory=(),
        budget=ControlBudget(
            paid_calls_used=1,
            paid_call_limit=MAX_AGENT_TURNS,
            elapsed_s=10.0,
            wall_time_limit_s=900.0,
        ),
    )


class FakeControlSession:
    """Zero-provider protocol preflight; it is never a training reward source."""

    def __init__(
        self,
        rollout_id: str,
        *,
        unified_control: bool = False,
        capability_refs: bool = False,
    ) -> None:
        self.rollout_id = rollout_id
        self.unified_control = bool(unified_control)
        self.capability_refs = bool(capability_refs)
        _control_protocol(
            unified_control=self.unified_control,
            capability_refs=self.capability_refs,
        )
        self.request_id = 0
        self.state = _fake_state(
            completion_requested=False,
            initial=self.unified_control,
        )
        self.done = False
        self.reward = 0.0
        self.error: str | None = None
        self.decisions: list[str] = []
        self.corrections = 0
        self.normalizations = 0
        self._pending_correction: str | None = None

    async def start(self) -> None:
        return None

    def _request(self) -> dict[str, Any]:
        builder = (
            build_control_action_messages
            if self.unified_control
            else build_control_decision_messages
        )
        messages, prompt_tokens, compacted = builder(
            self.state,
            capability_refs=self.capability_refs,
        )
        if self._pending_correction is not None:
            messages = [
                *messages,
                {"role": "user", "content": self._pending_correction},
            ]
        return {
            "version": BRIDGE_VERSION,
            "rollout_id": self.rollout_id,
            "request_id": self.request_id,
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "compacted": compacted,
            "correction": self._pending_correction,
            "state": asdict(self.state),
            "control_protocol": _control_protocol(
                unified_control=self.unified_control,
                capability_refs=self.capability_refs,
            ),
        }

    def _result(self) -> dict[str, Any]:
        metadata = {
            "grpo_bridge_version": BRIDGE_VERSION,
            "grpo_rollout_id": self.rollout_id,
            "grpo_external_live_controller": True,
            "grpo_control_protocol": _control_protocol(
                unified_control=self.unified_control,
                capability_refs=self.capability_refs,
            ),
            "grpo_control_requests": self.request_id,
            "grpo_control_responses": self.request_id,
            "worker_provider_base": YUNWU_API_BASE,
            "worker_models": list(DEFAULT_WORKER_MODELS),
            "provider_owner_retry_limit": 0,
            "provider_owner_retries": 0,
            "max_agent_turns": MAX_AGENT_TURNS,
            "fair_position_call_budget": None,
            "paid_worker_call_attempts": 0,
            "runtime_revision": PRODUCT_RUNTIME_REVISION,
            "workspace_snapshot_ready": False,
            "workspace_root": None,
            "live_control_failures": int(self.error is not None),
            "grpo_worker_timeout_s": 600.0,
            "protected_test_restore_policy": PROTECTED_TEST_POLICY,
            "protected_test_snapshot_entries": 1,
            "protected_test_restores": [],
        }
        return {
            "agent_result": {"metadata": metadata},
            "verifier_result": {"rewards": {"reward": self.reward}},
            "fake_protocol_preflight": True,
        }

    async def next_event(self, after_request_id: int) -> SessionEvent:
        if self.done:
            return SessionEvent(kind="result", result=self._result(), returncode=0)
        if after_request_id != self.request_id:
            raise RuntimeError("fake session request sequence mismatch")
        self.request_id += 1
        return SessionEvent(kind="request", request=self._request())

    async def submit(self, request_id: int, completion: str) -> None:
        if request_id != self.request_id:
            raise RuntimeError("fake session response sequence mismatch")
        try:
            if not self.unified_control:
                action = parse_control_decision(completion)
            elif self.capability_refs:
                action = parse_capability_control_action(
                    completion,
                    capability_reference_map(self.state.workers),
                )
            else:
                action = parse_control_action(completion)
            action, normalization = canonicalize_control_decision(action, self.state)
            self.normalizations += int(normalization is not None)
            if self.unified_control:
                validate_control_action(action, self.state)
            else:
                validate_control_decision(action, self.state)
        except ControlContractError as exc:
            if self.corrections < MAX_DECISION_CORRECTIONS:
                self.corrections += 1
                self._pending_correction = (
                    render_control_action_correction(
                        str(exc),
                        correction_attempt=self.corrections,
                    )
                    if self.unified_control
                    else render_decision_correction(
                        str(exc),
                        correction_attempt=self.corrections,
                    )
                )
                return
            self.error = f"{type(exc).__name__}: {exc}"
            self.done = True
            return
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.done = True
            return
        self._pending_correction = None
        self.decisions.append(action.action)
        if self.unified_control and self.decisions == ["replan"]:
            first = action.steps[0]
            self.state = _fake_state(
                completion_requested=False,
                worker_id=first.worker_id,
                subtask=first.subtask,
            )
            return
        if self.unified_control and self.decisions == ["replan", "continue"]:
            active = self.state.active_position
            assert active is not None
            self.state = replace(
                _fake_state(
                    completion_requested=True,
                    worker_id=active.worker_id,
                    subtask=active.subtask,
                ),
                budget=replace(self.state.budget, paid_calls_used=2),
            )
            return
        if self.unified_control:
            self.done = True
            self.reward = float(
                self.decisions == ["replan", "continue", "complete"]
            )
            return
        if self.decisions == ["continue"]:
            self.state = replace(
                _fake_state(completion_requested=True),
                budget=replace(self.state.budget, paid_calls_used=2),
            )
            return
        self.done = True
        self.reward = float(self.decisions == ["continue", "complete"])

    async def close(self) -> None:
        self.done = True


def _attest_result(
    result: dict[str, Any],
    *,
    rollout_id: str,
    returncode: int | None,
    allow_fake: bool,
    mode: str = "live",
    unified_control: bool = False,
    capability_refs: bool = False,
) -> dict[str, Any]:
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    paid_calls = metadata.get("paid_worker_call_attempts")
    requests = metadata.get("grpo_control_requests")
    responses = metadata.get("grpo_control_responses")
    fake_protocol = bool(result.get("fake_protocol_preflight", False))
    if mode == "local":
        provider_ok = (
            metadata.get("worker_provider_base") == LOCAL_PROVIDER_BASE
            and metadata.get("worker_calls_are_paid") is False
        )
        pool_ok = tuple(metadata.get("worker_models") or ()) == LOCAL_WORKER_MODELS
    else:
        provider_ok = metadata.get("worker_provider_base") == YUNWU_API_BASE
        pool_ok = tuple(metadata.get("worker_models") or ()) == DEFAULT_WORKER_MODELS
    checks = {
        "returncode": returncode == 0,
        "bridge_version": metadata.get("grpo_bridge_version") == BRIDGE_VERSION,
        "rollout_id": metadata.get("grpo_rollout_id") == rollout_id,
        "external_controller": metadata.get("grpo_external_live_controller") is True,
        "control_protocol": metadata.get("grpo_control_protocol")
        == _control_protocol(
            unified_control=unified_control,
            capability_refs=capability_refs,
        ),
        "provider": provider_ok,
        "pool": pool_ok,
        "retry_limit": metadata.get("provider_owner_retry_limit") == 0,
        "no_provider_retry": metadata.get("provider_owner_retries") == 0,
        "no_position_lease": metadata.get("fair_position_call_budget") is None,
        "global_call_limit": metadata.get("max_agent_turns") == MAX_AGENT_TURNS,
        "paid_calls": (
            isinstance(paid_calls, int)
            and not isinstance(paid_calls, bool)
            and 0 <= paid_calls <= MAX_AGENT_TURNS
        ),
        "control_round_trip": (
            isinstance(requests, int)
            and requests > 0
            and responses == requests
        ),
        "runtime_revision": metadata.get("runtime_revision")
        == PRODUCT_RUNTIME_REVISION,
        "workspace_snapshot": (
            metadata.get("workspace_snapshot_ready") is True
            and metadata.get("workspace_root") in {"/app", "/testbed"}
        ),
        "worker_timeout": (
            isinstance(metadata.get("grpo_worker_timeout_s"), (int, float))
            and not isinstance(metadata.get("grpo_worker_timeout_s"), bool)
            and float(metadata["grpo_worker_timeout_s"]) >= 600.0
        ),
        "protected_tests": (
            metadata.get("protected_test_restore_policy")
            == PROTECTED_TEST_POLICY
            and isinstance(metadata.get("protected_test_restores"), list)
            and (
                (
                    metadata.get("protected_test_repo") in {"/app", "/testbed"}
                    and isinstance(
                        metadata.get("protected_test_snapshot_entries"), int
                    )
                    and metadata.get("protected_test_snapshot_entries", 0) > 0
                )
                or (
                    mode == "local"
                    and metadata.get("protected_test_repo") is None
                    and metadata.get("protected_test_snapshot_entries") == 0
                )
            )
        ),
        "not_fake": not fake_protocol,
    }
    reward = _numeric_reward(result)
    failed = sorted(name for name, passed in checks.items() if not passed)
    trainable = reward is not None and not failed and not fake_protocol
    return {
        "reward": float(reward or 0.0) if trainable else 0.0,
        "trainable": trainable,
        "checks": checks,
        "failed_checks": failed,
        "paid_calls": paid_calls if isinstance(paid_calls, int) else 0,
        "control_requests": requests if isinstance(requests, int) else 0,
        "live_control_failures": int(metadata.get("live_control_failures") or 0),
        "fake_protocol_allowed": bool(allow_fake and fake_protocol),
    }


class FuguLiveAgenticGRPOEnv(vf.MultiTurnEnv):
    """Non-linear multi-turn rollout over independent production decision prompts."""

    def __init__(
        self,
        *,
        provider_mode: str,
        artifact_root: Path,
        event_timeout_s: float,
        worker_timeout_s: float,
        decision_timeout_s: float,
        max_parallel_sessions: int,
        unified_control: bool,
        capability_refs: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if provider_mode not in {"fake", "live", "local"}:
            raise ValueError("provider_mode must be fake, live, or local")
        if provider_mode == "live" and worker_timeout_s < 600:
            raise ValueError("live Yunwu worker timeout must be at least 600 seconds")
        if max_parallel_sessions <= 0:
            raise ValueError("max_parallel_sessions must be positive")
        self.provider_mode = provider_mode
        self.artifact_root = artifact_root
        self.event_timeout_s = event_timeout_s
        self.worker_timeout_s = worker_timeout_s
        self.decision_timeout_s = decision_timeout_s
        self.max_parallel_sessions = max_parallel_sessions
        self.unified_control = bool(unified_control)
        self.capability_refs = bool(capability_refs)
        _control_protocol(
            unified_control=self.unified_control,
            capability_refs=self.capability_refs,
        )
        self._session_slots = asyncio.Semaphore(max_parallel_sessions)
        self._sessions: dict[str, ControlSession] = {}
        self._fatal_setup_error: str | None = None
        self._group_fatal_error: str | None = None

    def _new_session(self, state: dict[str, Any]) -> ControlSession:
        rollout_id = str(state["trajectory_id"])
        if self.provider_mode == "fake":
            return FakeControlSession(
                rollout_id,
                unified_control=self.unified_control,
                capability_refs=self.capability_refs,
            )
        info = _info(state.get("info"))
        return HarborControlSession(
            rollout_id=rollout_id,
            task_dir=Path(info["task_dir"]),
            artifact_root=self.artifact_root,
            event_timeout_s=self.event_timeout_s,
            worker_timeout_s=self.worker_timeout_s,
            decision_timeout_s=self.decision_timeout_s,
            provider_mode=self.provider_mode,
            unified_control=self.unified_control,
            capability_refs=self.capability_refs,
        )

    @staticmethod
    def _failure_messages(error: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Return one compact JSON control decision and no prose.",
            },
            {
                "role": "user",
                "content": (
                    "The agentic environment failed before a live decision boundary. "
                    "Return {\"action\":\"replan\"}; this rollout will be excluded "
                    f"from training. Runtime error: {error[:1000]}"
                ),
            },
        ]

    @staticmethod
    def _set_request(state: dict[str, Any], request: dict[str, Any]) -> None:
        state["_control_request_id"] = int(request["request_id"])
        state["_control_messages"] = request["messages"]
        state["_last_prompt_tokens"] = request.get("prompt_tokens")
        state["_prompt_compactions"] = int(
            state.get("_prompt_compactions", 0)
        ) + int(bool(request.get("compacted")))

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state.update(
            {
                "_agentic_reward": 0.0,
                "_agentic_trainable": False,
                "_agentic_failed_checks": [],
                "_agentic_paid_calls": 0,
                "_agentic_control_requests": 0,
                "_prompt_compactions": 0,
            }
        )
        await self._session_slots.acquire()
        state["_agentic_session_slot_acquired"] = True
        group_abort_reason: str | None = None
        if self._fatal_setup_error is not None:
            group_abort_reason = (
                "group aborted after an earlier rollout failed before its first "
                f"control boundary: {self._fatal_setup_error}"
            )
        elif self._group_fatal_error is not None:
            group_abort_reason = (
                "group aborted after an earlier rollout produced a fatal "
                f"post-boundary result: {self._group_fatal_error}"
            )
        if group_abort_reason is not None:
            state["_agentic_setup_error"] = group_abort_reason
            state["_agentic_group_aborted"] = True
            state["_control_request_id"] = 0
            state["_control_messages"] = self._failure_messages(
                state["_agentic_setup_error"]
            )
            state["_agentic_session_slot_acquired"] = False
            self._session_slots.release()
            return state
        rollout_id = str(state["trajectory_id"])
        session: ControlSession | None = None
        try:
            session = self._new_session(state)
            self._sessions[rollout_id] = session
            await session.start()
            event = await session.next_event(0)
            if event.kind != "request" or event.request is None:
                raise RuntimeError(event.error or "task ended before first control boundary")
            self._set_request(state, event.request)
        except Exception as exc:
            state["_agentic_setup_error"] = f"{type(exc).__name__}: {exc}"
            self._fatal_setup_error = state["_agentic_setup_error"]
            state["_control_request_id"] = 0
            state["_control_messages"] = self._failure_messages(
                state["_agentic_setup_error"]
            )
            try:
                if session is not None:
                    await session.close()
            except Exception as close_exc:
                state["_agentic_setup_close_error"] = (
                    f"{type(close_exc).__name__}: {close_exc}"
                )
            finally:
                self._sessions.pop(rollout_id, None)
                state["_agentic_session_slot_acquired"] = False
                self._session_slots.release()
        return state

    async def get_prompt_messages(self, state: dict[str, Any]) -> Any:
        if not state["trajectory"]:
            return normalize_messages(
                state["_control_messages"], field_name="control_messages"
            )
        await self.env_response([], state)
        return normalize_messages(
            state["_control_messages"], field_name="control_messages"
        )

    async def env_response(
        self, messages: Any, state: dict[str, Any], **kwargs: Any
    ) -> Any:
        del messages, kwargs
        if state.get("final_env_response") is not None:
            return state["final_env_response"]
        if state.get("_agentic_setup_error"):
            state["final_env_response"] = [
                {"role": "user", "content": "Rollout excluded: environment setup failed."}
            ]
            return state["final_env_response"]
        rollout_id = str(state["trajectory_id"])
        session = self._sessions[rollout_id]
        if self.provider_mode == "live" and self._group_fatal_error is not None:
            state["_agentic_runtime_error"] = (
                "rollout stopped before its next decision: the group already has "
                f"a fatal result: {self._group_fatal_error}"
            )
            state["_agentic_group_aborted"] = True
            try:
                await session.close()
            finally:
                self._sessions.pop(rollout_id, None)
            state["final_env_response"] = [
                {"role": "user", "content": "Rollout excluded: group aborted."}
            ]
            return state["final_env_response"]
        request_id = int(state["_control_request_id"])
        completion = _completion_text(state["trajectory"][-1]["completion"])
        try:
            await session.submit(request_id, completion)
            event = await session.next_event(request_id)
            if event.kind == "request" and event.request is not None:
                self._set_request(state, event.request)
                return []
            if event.kind != "result" or event.result is None:
                raise RuntimeError(event.error or "Harbor session ended without a result")
            attestation = _attest_result(
                event.result,
                rollout_id=rollout_id,
                returncode=event.returncode,
                allow_fake=self.provider_mode == "fake",
                mode=self.provider_mode,
                unified_control=self.unified_control,
                capability_refs=self.capability_refs,
            )
            state["_agentic_reward"] = attestation["reward"]
            state["_agentic_trainable"] = attestation["trainable"]
            state["_agentic_failed_checks"] = attestation["failed_checks"]
            state["_agentic_paid_calls"] = attestation["paid_calls"]
            state["_agentic_control_requests"] = attestation["control_requests"]
            state["_agentic_live_control_failures"] = attestation[
                "live_control_failures"
            ]
            if self.provider_mode == "live" and not attestation["trainable"]:
                self._group_fatal_error = (
                    f"rollout {rollout_id} is untrainable: "
                    f"{attestation['failed_checks'] or ['missing functional reward']}"
                )
            state["final_env_response"] = [
                {
                    "role": "user",
                    "content": (
                        "Functional verifier reward: "
                        f"{attestation['reward']:.1f}. Rollout complete."
                    ),
                }
            ]
            return state["final_env_response"]
        except Exception as exc:
            state["_agentic_runtime_error"] = f"{type(exc).__name__}: {exc}"
            if self.provider_mode == "live":
                self._group_fatal_error = (
                    f"rollout {rollout_id} runtime failure: "
                    f"{state['_agentic_runtime_error']}"
                )
            state["final_env_response"] = [
                {"role": "user", "content": "Rollout excluded: runtime failure."}
            ]
            return state["final_env_response"]

    @vf.cleanup(priority=100)
    async def close_control_session(self, state: dict[str, Any]) -> None:
        rollout_id = str(state.get("trajectory_id", ""))
        session = self._sessions.pop(rollout_id, None)
        if session is not None:
            await session.close()
        if state.pop("_agentic_session_slot_acquired", False):
            self._session_slots.release()


def live_agentic_reward(state: dict[str, Any]) -> float:
    return float(state.get("_agentic_reward", 0.0))


def live_agentic_trainable(state: dict[str, Any]) -> float:
    return float(state.get("_agentic_trainable", False))


def live_agentic_success(state: dict[str, Any]) -> float:
    return float(state.get("_agentic_reward", 0.0) >= 1.0)


def live_agentic_paid_calls(state: dict[str, Any]) -> float:
    return float(state.get("_agentic_paid_calls", 0))


def live_agentic_control_requests(state: dict[str, Any]) -> float:
    return float(state.get("_agentic_control_requests", 0))


def live_agentic_prompt_compactions(state: dict[str, Any]) -> float:
    return float(state.get("_prompt_compactions", 0))


def _dataset(
    rows: list[dict[str, Any]],
    *,
    task_name: str,
    shuffle: bool,
    seed: int,
    max_examples: int | None,
) -> Dataset:
    selected = list(rows)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if max_examples is not None and max_examples > 0:
        selected = selected[:max_examples]
    dataset_rows = []
    for index, row in enumerate(selected):
        dataset_rows.append(
            {
                "example_id": index,
                "task": task_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": "Await the live product control state.",
                    },
                    {
                        "role": "user",
                        "content": "The environment will replace this with a live state.",
                    },
                ],
                "answer": row["task_id"],
                "info": json.dumps(
                    {
                        "task_id": row["task_id"],
                        "task_dir": row["task_dir"],
                        "task_tree_sha256": row["task_tree_sha256"],
                    },
                    sort_keys=True,
                ),
            }
        )
    return Dataset.from_list(dataset_rows)


def load_environment(
    task_manifest_path: str = str(DEFAULT_MANIFEST),
    task_name: str = "fugu_live_agentic_grpo",
    provider_mode: str = "fake",
    allow_yunwu_live: bool = False,
    artifact_root: str = str(DEFAULT_ARTIFACT_ROOT),
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 20260717,
    worker_timeout_s: float = 600.0,
    decision_timeout_s: float = 180.0,
    event_timeout_s: float = 1800.0,
    rollout_timeout_s: float = 2400.0,
    max_control_decisions: int = MAX_CONTROL_DECISIONS,
    max_parallel_sessions: int = 1,
    unified_control: bool = False,
    capability_refs: bool = False,
    score_rollouts: bool = True,
    max_seq_len: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load the bounded product-runtime GRPO environment.

    Live mode is fail-closed and requires an explicit Yunwu opt-in. Fake mode
    exercises only the bridge and never calls a provider.
    """
    if provider_mode == "live" and not allow_yunwu_live:
        raise ValueError("live GRPO requires allow_yunwu_live=true")
    if provider_mode == "live" and not os.environ.get("YUNWU_API_KEY"):
        raise ValueError("YUNWU_API_KEY is required for live GRPO")
    _control_protocol(
        unified_control=unified_control,
        capability_refs=capability_refs,
    )
    if max_control_decisions <= 0:
        raise ValueError("max_control_decisions must be positive")
    if max_control_decisions > MAX_CONTROL_DECISIONS:
        raise ValueError(
            f"max_control_decisions cannot exceed {MAX_CONTROL_DECISIONS}"
        )
    if provider_mode == "live" and max_control_decisions != MAX_CONTROL_DECISIONS:
        raise ValueError(
            "live GRPO must preserve the complete 120-call lifecycle with "
            f"max_control_decisions={MAX_CONTROL_DECISIONS}"
        )
    rows = _read_manifest(Path(task_manifest_path).expanduser().resolve())
    dataset = _dataset(
        rows,
        task_name=task_name,
        shuffle=shuffle,
        seed=seed,
        max_examples=max_examples,
    )
    rubric = vf.Rubric(
        funcs=[
            live_agentic_reward,
            live_agentic_trainable,
            live_agentic_success,
            live_agentic_paid_calls,
            live_agentic_control_requests,
            live_agentic_prompt_compactions,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return FuguLiveAgenticGRPOEnv(
        provider_mode=provider_mode,
        artifact_root=Path(artifact_root).expanduser().resolve(),
        event_timeout_s=event_timeout_s,
        worker_timeout_s=worker_timeout_s,
        decision_timeout_s=decision_timeout_s,
        max_parallel_sessions=max_parallel_sessions,
        unified_control=unified_control,
        capability_refs=capability_refs,
        max_turns=max_control_decisions,
        timeout_seconds=rollout_timeout_s,
        dataset=dataset,
        rubric=rubric,
        score_rollouts=score_rollouts,
        max_seq_len=max_seq_len,
        env_args={
            "version": ENV_VERSION,
            "task_manifest_path": task_manifest_path,
            "provider_mode": provider_mode,
            "allow_yunwu_live": allow_yunwu_live,
            "worker_timeout_s": worker_timeout_s,
            "max_paid_calls": MAX_AGENT_TURNS,
            "max_parallel_sessions": max_parallel_sessions,
            "control_protocol": _control_protocol(
                unified_control=unified_control,
                capability_refs=capability_refs,
            ),
            "pre_boundary_group_abort": True,
            "provider": YUNWU_API_BASE if provider_mode == "live" else None,
        },
        **kwargs,
    )
