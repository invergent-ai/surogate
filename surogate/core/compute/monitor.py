"""Background monitor for SkyPilot serving services and managed jobs.

Runs a single asyncio task that bulk-fetches status from SkyPilot every
``poll_interval`` seconds, diffs against the DB, batch-updates changed rows,
and fires registered callbacks on status transitions.
"""

import asyncio
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from surogate.core.compute.sky import (
    _get_managed_job_statuses,
    _get_serve_statuses,
    _map_status,
)
from surogate.core.db.engine import get_session_factory
from surogate.core.db.repository import compute as repo
from surogate.utils.logger import get_logger

if TYPE_CHECKING:
    from surogate.core.compute.local_tasks import LocalTaskManager

logger = get_logger()

# Type alias for transition callbacks (async):
#   (entity_type, entity_id, name, old_status, new_status, data)
TransitionCallback = Callable[[str, str, str, str, str, dict[str, Any]], Awaitable[None]]


class ServingMonitor:
    """Periodically polls SkyPilot for status changes and updates the DB.

    Also wraps ``LocalTaskManager._watch`` so that local-task completions
    flow through the same transition callback system.

    Usage::

        monitor = ServingMonitor(poll_interval=5, task_manager=tm)
        monitor.on_transition(my_alert_handler)
        await monitor.start()   # spawns background task
        ...
        await monitor.stop()    # cancels on shutdown
    """

    def __init__(
        self,
        poll_interval: float = 5.0,
        task_manager: Optional["LocalTaskManager"] = None,
    ) -> None:
        self._poll_interval = poll_interval
        self._task: Optional[asyncio.Task] = None
        self._callbacks: list[TransitionCallback] = []
        self._task_manager = task_manager

        # Wire ourselves as the watch-creator so every spawned local task
        # fires transition callbacks when it completes.
        if task_manager is not None:
            task_manager._create_watch = self._create_local_task_watch

    # ── Public API ──────────────────────────────────────────────────

    def on_transition(self, cb: TransitionCallback) -> None:
        """Register a callback invoked on every status transition.

        The callback receives
        ``(entity_type, entity_id, name, old_status, new_status, data)``.
        ``entity_type`` is ``"model"``, ``"job"``, or ``"task"``.
        ``data`` is the full entity dict (same shape as the REST response).
        """
        self._callbacks.append(cb)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("ServingMonitor started (interval=%ss)", self._poll_interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("ServingMonitor stopped")

    # ── Internal ────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("ServingMonitor tick failed", exc_info=True)
            await asyncio.sleep(self._poll_interval)

    async def _tick(self) -> None:
        factory = get_session_factory()

        # 1. Bulk-fetch from SkyPilot (2 calls total, regardless of count)
        sky_serve, sky_jobs = await asyncio.gather(
            _get_serve_statuses(),
            _get_managed_job_statuses(),
            return_exceptions=True,
        )
        if isinstance(sky_serve, Exception):
            logger.debug("Could not fetch SkyPilot serve statuses: %s", sky_serve)
            sky_serve = {}
        if isinstance(sky_jobs, Exception):
            logger.debug("Could not fetch SkyPilot job statuses: %s", sky_jobs)
            sky_jobs = {}

        # 2. Load only active (non-terminal) records from DB
        async with factory() as session:
            active_services = await repo.list_active_serving_services(session)
            active_jobs = await repo.list_active_managed_jobs(session)

            # 3. Diff and update serving services
            await self._sync_serving_services(session, active_services, sky_serve)

            # 4. Diff and update managed jobs
            await self._sync_managed_jobs(session, active_jobs, sky_jobs)

    async def _sync_serving_services(
        self, session, db_services, sky_statuses: dict,
    ) -> None:
        from surogate.core.compute.models import build_model_response

        now = datetime.now(timezone.utc)

        for svc in db_services:
            if svc.name not in sky_statuses:
                continue

            info = sky_statuses[svc.name]
            new_status = info.get("status", "")
            if new_status == svc.status:
                continue

            old_status = svc.status

            # Build update dict
            updates: dict = {"status": new_status}
            if new_status == "ready" and svc.started_at is None:
                updates["started_at"] = now
            if new_status in ("failed", "failed_cleanup", "controller_failed"):
                updates["terminated_at"] = now

            # Endpoint is managed by our proxy (set at launch time).
            # Don't overwrite with SkyPilot's stale localhost:<lb_port>.

            await repo.update_serving_service(session, svc.id, **updates)
            # Apply updates in-memory so the response builder sees fresh data
            for k, v in updates.items():
                setattr(svc, k, v)

            logger.info(
                "ServingService %s (%s): %s -> %s",
                svc.name, svc.id, old_status, new_status,
            )

            # Build full DeployedModelResponse for the linked model
            model = await repo.get_deployed_model_by_service(session, svc.id)
            if model is not None:
                resp = build_model_response(model, svc)
                data = resp.model_dump(mode="json")
                await self._fire_callbacks("model", model.id, model.name, old_status, new_status, data)

    async def _sync_managed_jobs(
        self, session, db_jobs, sky_statuses: dict[int, str],
    ) -> None:
        now = datetime.now(timezone.utc)

        for job in db_jobs:
            if job.skypilot_job_id is None:
                continue
            if job.skypilot_job_id not in sky_statuses:
                continue

            new_status = _map_status(sky_statuses[job.skypilot_job_id])
            if new_status == job.status:
                continue

            old_status = job.status

            started = now if new_status == "running" and job.started_at is None else None
            completed = now if new_status in ("completed", "failed", "cancelled") else None

            await repo.update_managed_job_status(
                session, job.id, new_status,
                started_at=started, completed_at=completed,
            )

            logger.info(
                "ManagedJob %s (%s): %s -> %s",
                job.name, job.id, old_status, new_status,
            )

            data = _job_to_dict(job, new_status, started)
            await self._fire_callbacks("job", job.id, job.name, old_status, new_status, data)

    # ── Local task watch wrapper ─────────────────────────────────────

    def _create_local_task_watch(
        self, task_id: str, name: str,
        proc: subprocess.Popen, log_file,
    ) -> None:
        """Called by LocalTaskManager.spawn() instead of creating the watch directly."""
        asyncio.create_task(self._watch_local_task(task_id, name, proc, log_file))

    async def _watch_local_task(
        self, task_id: str, name: str,
        proc: subprocess.Popen, log_file,
    ) -> None:
        """Delegate to LocalTaskManager._watch, then fire transition callbacks."""
        await self._task_manager._watch(task_id, proc, log_file)

        new_status = "completed" if proc.returncode == 0 else "failed"

        # Load fresh record from DB so the response matches the REST shape
        factory = get_session_factory()
        async with factory() as session:
            data = await _load_task_dict(session, task_id)

        await self._fire_callbacks("task", task_id, name, "running", new_status, data)

    # ── Fire callbacks ──────────────────────────────────────────────

    async def _fire_callbacks(
        self, entity_type: str, entity_id: str, name: str,
        old_status: str, new_status: str, data: dict[str, Any],
    ) -> None:
        for cb in self._callbacks:
            try:
                await cb(entity_type, entity_id, name, old_status, new_status, data)
            except Exception:
                logger.warning(
                    "Transition callback failed for %s %s", entity_type, name,
                    exc_info=True,
                )


# ── Entity serialisers (same shape as REST responses) ──────────────


def _job_to_dict(job, status: str, started_at=None) -> dict[str, Any]:
    """Build a JobResponse-shaped dict from a ManagedJob ORM object."""
    return {
        "id": job.id,
        "name": job.name,
        "type": job.workload_type.value,
        "method": "\u2014",
        "status": status,
        "gpu": job.accelerators,
        "gpu_count": 0,
        "location": job.cloud or "local",
        "node": None,
        "eta": None,
        "started_at": (started_at or job.started_at).isoformat() if (started_at or job.started_at) else None,
        "requested_by": job.requested_by_id,
        "project": job.project_id,
        "cloud": job.cloud,
        "region": job.region,
        "use_spot": job.use_spot,
        "skypilot_job_id": job.skypilot_job_id,
    }


async def _load_task_dict(session, task_id: str) -> dict[str, Any]:
    """Load a LocalTask and build a LocalTaskResponse-shaped dict."""
    import sqlalchemy as sa
    from surogate.core.db.models.compute import LocalTask

    result = await session.execute(
        sa.select(LocalTask).where(LocalTask.id == task_id)
    )
    t = result.scalar_one_or_none()
    if t is None:
        return {}
    return {
        "id": t.id,
        "name": t.name,
        "task_type": t.task_type.value,
        "status": t.status.value,
        "pid": t.pid,
        "exit_code": t.exit_code,
        "error_message": t.error_message,
        "progress": t.progress,
        "project_id": t.project_id,
        "requested_by": t.requested_by_id,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
    }
