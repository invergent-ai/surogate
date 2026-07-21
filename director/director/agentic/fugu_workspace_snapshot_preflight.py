"""Zero-provider Harbor agent for the product workspace-snapshot boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from director.agentic.fugu_ultra_terminal import (
    PRODUCT_RUNTIME_REVISION,
    WORKSPACE_ROOT,
    WORKSPACE_SNAPSHOT_ROOT,
    FuguUltraTerminalAgent,
)
from director.agentic.prepared_index_test_protection import (
    PreparedIndexTestProtectionMixin,
)


class FuguWorkspaceSnapshotPreflightAgent(
    PreparedIndexTestProtectionMixin,
    FuguUltraTerminalAgent,
):
    """Exercise the exact production snapshot code without invoking any model."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs: object,
    ) -> None:
        self._initialize_protected_test_protection()
        kwargs.setdefault("provider_owner_retry_limit", 0)
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    @override
    def name() -> str:
        return "fugu-workspace-snapshot-preflight"

    @override
    def version(self) -> str | None:
        return f"workspace-snapshot-preflight-{PRODUCT_RUNTIME_REVISION}"

    async def _audit_workspace_isolation(
        self, environment: BaseEnvironment
    ) -> dict[str, Any] | None:
        """Optional subclass hook for stronger task-specific isolation checks."""
        del environment
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._fugu_llm.reset_for_run()
        self._protected_test_environment = environment
        self._protected_test_restores.clear()
        self._untracked_test_removals.clear()
        self._protected_test_snapshot.clear()
        self._protected_test_repo = None
        self._prepared_repository_setup_executed = False
        self._prepared_git_history_sanitization = None
        self._active_environment = environment
        self._workspace_snapshot_ready = False
        self._workspace_root = WORKSPACE_ROOT
        self._workspace_snapshot_token = ""
        self._workspace_snapshot_summary = {}
        self._workspace_integrity_checks = 0
        self._workspace_recoveries = 0
        self._workspace_recovery_failures = 0
        self._workspace_cleanup_failures = 0
        workspace_isolation = None
        try:
            task_instruction = await self._prepare_protected_repository(instruction)
            self._fugu_llm.set_task_instruction(task_instruction)
            await self._prepare_workspace_snapshot(environment)
            workspace_isolation = await self._audit_workspace_isolation(environment)
        finally:
            try:
                await self._remove_workspace_sentinel()
            finally:
                self._record_fugu_metadata(context)
                metadata = dict(context.metadata or {})
                metadata.update(
                    {
                        **self._protected_test_metadata(),
                        "workspace_snapshot_preflight": True,
                        "workspace_snapshot_preflight_provider_calls": 0,
                        "workspace_snapshot_preflight_paid_calls": 0,
                        "workspace_isolation": workspace_isolation,
                    }
                )
                context.metadata = metadata
                self._active_environment = None


class FuguSanitizedWorkspaceSnapshotPreflightAgent(
    FuguWorkspaceSnapshotPreflightAgent
):
    """Zero-provider preflight including gold-history isolation."""

    _sanitize_prepared_git_history = True

    @staticmethod
    @override
    def name() -> str:
        return "fugu-sanitized-workspace-snapshot-preflight"

    @override
    async def _audit_workspace_isolation(
        self, environment: BaseEnvironment
    ) -> dict[str, Any]:
        expected_head = (
            (self._prepared_git_history_sanitization or {}).get("head")
        )
        if not isinstance(expected_head, str) or not expected_head:
            raise RuntimeError("workspace isolation audit has no sanitized Git head")
        script = f"""python3 - <<'PY'
import json
import pathlib
import subprocess

forbidden = (
    '/solution',
    '/tests',
    '/test_patch.diff',
    '/config.json',
)
visible = [path for path in forbidden if pathlib.Path(path).exists()]

def run(*args):
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )

repositories = []
for repo in ({json.dumps(self._workspace_root)}, {json.dumps(WORKSPACE_SNAPSHOT_ROOT + '/original')}):
    head = run('git', '-C', repo, 'rev-parse', 'HEAD')
    count = run('git', '-C', repo, 'rev-list', '--all', '--count')
    remotes = run('git', '-C', repo, 'remote')
    if head.returncode or count.returncode or remotes.returncode:
        raise SystemExit('failed to inspect isolated repository ' + repo)
    repositories.append({{
        'repo': repo,
        'head': head.stdout.strip(),
        'commit_count': int(count.stdout.strip()),
        'remotes': [line for line in remotes.stdout.splitlines() if line],
    }})

print(json.dumps({{
    'forbidden_paths_checked': list(forbidden),
    'visible_forbidden_paths': visible,
    'repositories': repositories,
}}, sort_keys=True))
PY"""
        result = await environment.exec(script, cwd="/", timeout_sec=30, user="root")
        if result.return_code != 0:
            raise RuntimeError(
                "workspace isolation audit failed closed: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        payload = json.loads(result.stdout or "{}")
        repositories = payload.get("repositories")
        if (
            payload.get("visible_forbidden_paths") != []
            or not isinstance(repositories, list)
            or len(repositories) != 2
            or any(
                not isinstance(row, dict)
                or row.get("head") != expected_head
                or row.get("commit_count") != 1
                or row.get("remotes") != []
                for row in repositories
            )
        ):
            raise RuntimeError(
                "workspace isolation invariant failed: "
                + json.dumps(payload, ensure_ascii=True, sort_keys=True)
            )
        return payload
