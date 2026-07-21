"""Exact-environment live branchpoint collection.

This revision extends the workspace snapshot collector with a host handshake.
The prefix agent remains quiescent after exporting the branchpoint until the
campaign runner commits and attests the running task container. Branch arms can
therefore start from the exact installed environment, not a reconstructed one.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from pathlib import Path
from typing import Any, override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_live_branchpoint_collection import (
    BRANCHPOINT_PATH_ENV,
    BRANCH_ACTION_ENV,
    COLLECTION_ID_ENV,
    INITIAL_WORKFLOW_ENV,
    MODE_ENV,
    POOL_BINDING_ENV,
    FuguLiveBranchpointCollectionAgent,
    _sha256,
    _write_json_atomic,
)


COLLECTION_REVISION = "20260721-live-branchpoint-exact-environment-v2-test-namespace-guard-r2"
ENVIRONMENT_ACK_FILENAME = "environment_capture_ack.json"
ENVIRONMENT_CAPTURE_TIMEOUT_S = 300.0


class FuguLiveBranchpointCollectionAgentV2(FuguLiveBranchpointCollectionAgent):
    """Freeze the exact running task image at the common live branchpoint."""

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._environment_capture_ack = self._branchpoint_path.parent / ENVIRONMENT_ACK_FILENAME
        self._environment_capture_acknowledged = False

    @staticmethod
    @override
    def name() -> str:
        return "fugu-live-branchpoint-collection-v2"

    @override
    def version(self) -> str | None:
        return COLLECTION_REVISION

    async def _export_environment_inventory(self) -> None:
        environment = self._active_environment
        if environment is None:
            raise RuntimeError("active environment is unavailable during branchpoint capture")
        artifact_dir = "/logs/artifacts/live-branchpoint"
        freeze_path = f"{artifact_dir}/pip-freeze.txt"
        check_path = f"{artifact_dir}/pip-check.txt"
        python_path = f"{artifact_dir}/python-version.txt"
        command = f"""set -eu
mkdir -p {shlex.quote(artifact_dir)}
python -m pip freeze --all > {shlex.quote(freeze_path)}
python -VV > {shlex.quote(python_path)} 2>&1
set +e
python -m pip check > {shlex.quote(check_path)} 2>&1
pip_check_status=$?
set -e
printf 'pip_freeze_sha256='; sha256sum {shlex.quote(freeze_path)} | cut -d' ' -f1
printf 'pip_check_sha256='; sha256sum {shlex.quote(check_path)} | cut -d' ' -f1
printf 'python_version_sha256='; sha256sum {shlex.quote(python_path)} | cut -d' ' -f1
printf 'pip_check_status=%s\n' "$pip_check_status"
"""
        result = await environment.exec(command, cwd="/", timeout_sec=600, user="root")
        if result.return_code != 0:
            raise RuntimeError(f"failed to inventory branchpoint environment: {result.stderr}")
        values: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        required = {
            "pip_freeze_sha256",
            "pip_check_sha256",
            "python_version_sha256",
            "pip_check_status",
        }
        if required - values.keys():
            raise RuntimeError("environment inventory is incomplete")
        payload = json.loads(self._branchpoint_path.read_text(encoding="utf-8"))
        payload["environment"] = {
            "capture_protocol": "quiescent_host_container_commit_v1",
            "pip_freeze_path": freeze_path,
            "pip_freeze_sha256": values["pip_freeze_sha256"],
            "pip_check_path": check_path,
            "pip_check_sha256": values["pip_check_sha256"],
            "pip_check_status": int(values["pip_check_status"]),
            "python_version_path": python_path,
            "python_version_sha256": values["python_version_sha256"],
            "host_ack_filename": ENVIRONMENT_ACK_FILENAME,
        }
        _write_json_atomic(self._branchpoint_path, payload)

    async def _wait_for_host_environment_capture(self) -> None:
        deadline = time.monotonic() + ENVIRONMENT_CAPTURE_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._environment_capture_ack.is_file():
                ack = json.loads(self._environment_capture_ack.read_text(encoding="utf-8"))
                snapshot = json.loads(self._branchpoint_path.read_text(encoding="utf-8"))
                workspace = snapshot.get("workspace") or {}
                if ack.get("version") != "fugu_live_branchpoint_environment_capture_ack_v1":
                    raise RuntimeError("unsupported environment capture acknowledgement")
                if ack.get("branchpoint_sha256") != _sha256(self._branchpoint_path):
                    raise RuntimeError("environment capture acknowledged another branchpoint")
                if ack.get("workspace_sha256") != workspace.get("sha256"):
                    raise RuntimeError("environment capture acknowledged another workspace")
                image_id = ack.get("image_id")
                if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
                    raise RuntimeError("environment capture image ID is invalid")
                self._environment_capture_acknowledged = True
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("host did not commit the quiescent branchpoint environment")

    @override
    async def _export_workspace(self) -> None:
        await super()._export_workspace()
        controller = self._capture_controller
        if (
            controller is None
            or not controller.workspace_exported
            or self._environment_capture_acknowledged
        ):
            return
        payload = json.loads(self._branchpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("environment"), dict):
            await self._export_environment_inventory()
        await self._wait_for_host_environment_capture()

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "live_branchpoint_environment_captured": self._environment_capture_acknowledged,
                "live_branchpoint_environment_ack_path": str(self._environment_capture_ack),
                "live_branchpoint_environment_ack_sha256": (
                    _sha256(self._environment_capture_ack)
                    if self._environment_capture_ack.is_file()
                    else None
                ),
            }
        )
        context.metadata = metadata


__all__ = [
    "BRANCHPOINT_PATH_ENV",
    "BRANCH_ACTION_ENV",
    "COLLECTION_ID_ENV",
    "COLLECTION_REVISION",
    "ENVIRONMENT_ACK_FILENAME",
    "FuguLiveBranchpointCollectionAgentV2",
    "INITIAL_WORKFLOW_ENV",
    "MODE_ENV",
    "POOL_BINDING_ENV",
]
