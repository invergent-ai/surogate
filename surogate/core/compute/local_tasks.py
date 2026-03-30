"""Local task manager — spawns and tracks subprocess-based tasks.

Each task runs as a separate Python process (memory/IO isolation).
Stdout is captured to a log file; PROGRESS: and ERROR: lines are
parsed for status updates.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.compute import (
    LocalTask,
    LocalTaskStatus,
    LocalTaskType,
)
from surogate.utils.logger import get_logger

logger = get_logger()

TASKS_LOG_DIR = Path.home() / ".surogate" / "tasks"

# Map task types to the Python module that implements them.
TASK_MODULES = {
    LocalTaskType.import_model: "surogate.server.tasks.download_hf_model",
    LocalTaskType.import_dataset: "surogate.server.tasks.download_hf_dataset",
    # Future:
    # LocalTaskType.local_inference: "surogate.server.tasks.local_inference",
    # LocalTaskType.local_training: "surogate.server.tasks.local_training",
}


class LocalTaskManager:
    """Manages subprocess-based local tasks."""

    def __init__(self, config):
        self._config = config
        self._active: dict[str, subprocess.Popen] = {}
        TASKS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Spawn ────────────────────────────────────────────────────────

    async def spawn(
        self,
        session: AsyncSession,
        *,
        task_type: str,
        name: str,
        project_id: str,
        user_id: str,
        params: dict,
    ) -> LocalTask:
        """Create a DB record and launch the task subprocess.

        For import tasks, the LakeFS repository is created first (with
        proper auth/groups/policies via core.hub.lakefs) so the subprocess
        only needs to download + upload + commit.
        """
        ttype = LocalTaskType(task_type)
        module = TASK_MODULES.get(ttype)
        if module is None:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Pre-create LakeFS repo for import tasks
        if ttype in (LocalTaskType.import_model, LocalTaskType.import_dataset):
            await self._ensure_lakefs_repo(session, user_id, params, ttype)

        task = LocalTask(
            name=name,
            task_type=ttype,
            status=LocalTaskStatus.running,
            project_id=project_id,
            requested_by_id=user_id,
            params=json.dumps(params),
            started_at=datetime.utcnow(),
        )
        session.add(task)
        await session.flush()  # get task.id

        log_path = TASKS_LOG_DIR / f"{task.id}.log"
        task.log_path = str(log_path)

        env = await self._build_env(session, ttype, user_id, params)
        log_file = open(log_path, "w")

        proc = subprocess.Popen(
            [sys.executable, "-m", module],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        task.pid = proc.pid
        await session.commit()

        self._active[task.id] = proc
        logger.info("Task %s spawned (pid=%d, type=%s)", task.id, proc.pid, task_type)

        # Background watcher that updates DB when process exits
        asyncio.create_task(self._watch(task.id, proc, log_file))
        return task

    # ── Cancel ───────────────────────────────────────────────────────

    async def cancel(self, session: AsyncSession, task_id: str) -> None:
        """Send SIGTERM to a running task, SIGKILL after 5s."""
        proc = self._active.get(task_id)
        if proc is None:
            # Fallback: try to kill by PID from DB
            result = await session.execute(
                sa.select(LocalTask).where(LocalTask.id == task_id)
            )
            task = result.scalar_one_or_none()
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            if task.pid and _pid_alive(task.pid):
                os.kill(task.pid, signal.SIGTERM)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        await session.execute(
            sa.update(LocalTask)
            .where(LocalTask.id == task_id)
            .values(
                status=LocalTaskStatus.cancelled,
                completed_at=datetime.utcnow(),
            )
        )
        await session.commit()
        self._active.pop(task_id, None)

    # ── Logs ─────────────────────────────────────────────────────────

    async def get_logs(self, task_id: str, tail: int = 100) -> list[str]:
        """Read last N lines from the task's log file."""
        log_path = TASKS_LOG_DIR / f"{task_id}.log"
        if not log_path.exists():
            return []
        lines = log_path.read_text().splitlines()
        return lines[-tail:]

    # ── Reap (startup recovery) ──────────────────────────────────────

    async def reap(self, session: AsyncSession) -> None:
        """Clean up stale running tasks from a previous server crash."""
        result = await session.execute(
            sa.select(LocalTask).where(
                LocalTask.status == LocalTaskStatus.running
            )
        )
        stale = list(result.scalars().all())
        for task in stale:
            alive = task.pid is not None and _pid_alive(task.pid)
            if not alive:
                error_msg = _extract_error(task.log_path)
                await session.execute(
                    sa.update(LocalTask)
                    .where(LocalTask.id == task.id)
                    .values(
                        status=LocalTaskStatus.failed,
                        error_message=error_msg or "Process exited (server restarted)",
                        completed_at=datetime.utcnow(),
                    )
                )
        await session.commit()
        if stale:
            logger.info("Reaped %d stale tasks", len(stale))

    # ── Internal ─────────────────────────────────────────────────────

    async def _watch(
        self,
        task_id: str,
        proc: subprocess.Popen,
        log_file,
    ) -> None:
        """Poll subprocess until exit, then update DB."""
        from surogate.core.db.engine import get_session_factory

        while proc.poll() is None:
            await asyncio.sleep(2)

        log_file.close()
        exit_code = proc.returncode
        self._active.pop(task_id, None)

        # Determine final status
        if exit_code == 0:
            status = LocalTaskStatus.completed
        else:
            status = LocalTaskStatus.failed

        error_msg = _extract_error(str(TASKS_LOG_DIR / f"{task_id}.log"))
        progress = _extract_last_progress(str(TASKS_LOG_DIR / f"{task_id}.log"))

        factory = get_session_factory()
        async with factory() as session:
            await session.execute(
                sa.update(LocalTask)
                .where(LocalTask.id == task_id)
                .values(
                    status=status,
                    exit_code=exit_code,
                    error_message=error_msg,
                    progress=progress,
                    completed_at=datetime.utcnow(),
                )
            )
            await session.commit()

        logger.info(
            "Task %s finished (exit_code=%d, status=%s)",
            task_id,
            exit_code,
            status.value,
        )

    async def _ensure_lakefs_repo(
        self,
        session: AsyncSession,
        user_id: str,
        params: dict,
        task_type: LocalTaskType,
    ) -> None:
        """Create the LakeFS repo (with auth/groups/policies) before spawn."""
        from lakefs_sdk import RepositoryCreation
        from surogate.core.hub.lakefs import (
            create_repository,
            get_lakefs_client,
            REPO_TYPE_MODEL,
            REPO_TYPE_DATASET,
        )

        repo_id = params.get("lakefs_repo_id", "")
        if not repo_id:
            raise ValueError("lakefs_repo_id is required")

        repo_type = (
            REPO_TYPE_MODEL
            if task_type == LocalTaskType.import_model
            else REPO_TYPE_DATASET
        )
        client = await get_lakefs_client(user_id, session, self._config)
        req = RepositoryCreation(
            name=repo_id,
            storage_namespace=f"local://{repo_id}",
            default_branch=params.get("lakefs_branch", "main"),
            metadata={
                "type": repo_type,
                "externalId": params.get("hf_repo_id", ""),
            },
        )
        await create_repository(client, user_id, req, self._config)

    async def _build_env(
        self,
        session: AsyncSession,
        task_type: LocalTaskType,
        user_id: str,
        params: dict,
    ) -> dict:
        """Construct environment variables for the subprocess."""
        import surogate.core.db.repository.user as user_repo

        env = os.environ.copy()

        cfg = self._config
        if cfg.lakefs_endpoint:
            env["LAKECTL_SERVER_ENDPOINT_URL"] = cfg.lakefs_endpoint

        # Use per-user LakeFS credentials so the subprocess operates as that user
        creds = await user_repo.get_lakefs_credentials(session, user_id)
        if creds and all(creds):
            env["LAKECTL_CREDENTIALS_ACCESS_KEY_ID"] = creds[0]
            env["LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY"] = creds[1]

        # Task-specific params → env vars
        if task_type in (LocalTaskType.import_model, LocalTaskType.import_dataset):
            env["HF_REPO_ID"] = params.get("hf_repo_id", "")
            env["LAKEFS_REPO_ID"] = params.get("lakefs_repo_id", "")
            env["LAKEFS_BRANCH"] = params.get("lakefs_branch", "main")
            if params.get("hf_token"):
                env["HF_TOKEN"] = params["hf_token"]
            if params.get("hf_dataset_subset"):
                env["HF_DATASET_SUBSET"] = params["hf_dataset_subset"]

        return env


# ── Helpers ──────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _extract_error(log_path: Optional[str]) -> Optional[str]:
    """Extract the last ERROR: line from a log file."""
    if not log_path or not Path(log_path).exists():
        return None
    for line in reversed(Path(log_path).read_text().splitlines()):
        if line.startswith("ERROR:"):
            return line[len("ERROR:"):].strip()
    return None


def _extract_last_progress(log_path: Optional[str]) -> Optional[str]:
    """Extract the last PROGRESS: line from a log file."""
    if not log_path or not Path(log_path).exists():
        return None
    for line in reversed(Path(log_path).read_text().splitlines()):
        if line.startswith("PROGRESS:"):
            return line[len("PROGRESS:"):].strip()
    return None
