"""Compute service — wraps SkyPilot internal functions + Surogate DB.

SkyPilot is used as a direct library.  All blocking SkyPilot calls are
wrapped in ``asyncio.to_thread()`` so they don't block the event loop.

Implementation layer
  - sky.execution.launch / exec
  - sky.core.status / stop / down / cancel / cost_report
  - sky.jobs.server.core.launch   (managed jobs)
"""

import asyncio
from datetime import datetime
from typing import Optional

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.compute import CloudWorkloadType
from surogate.core.db.repository import compute as repo
from surogate.utils.logger import get_logger

logger = get_logger()

# ── Status mapping ───────────────────────────────────────────────────

_SKY_STATUS_MAP = {
    "PENDING": "queued",
    "SUBMITTED": "queued",
    "STARTING": "provisioning",
    "RUNNING": "running",
    "RECOVERING": "recovering",
    "CANCELLING": "cancelling",
    "SUCCEEDED": "completed",
    "CANCELLED": "cancelled",
    "FAILED": "failed",
    "FAILED_SETUP": "failed",
    "FAILED_PRECHECKS": "failed",
    "FAILED_NO_RESOURCE": "failed",
    "FAILED_CONTROLLER": "failed",
}


def _map_status(sky_status: str) -> str:
    return _SKY_STATUS_MAP.get(str(sky_status).upper(), "unknown")


# ── Managed Jobs ─────────────────────────────────────────────────────


async def launch_job(
    session: AsyncSession,
    *,
    task_yaml: str,
    name: str,
    project_id: str,
    workload_type: str,
    requested_by_id: str,
    accelerators: Optional[str] = None,
    cloud: Optional[str] = None,
    use_spot: bool = False,
):
    """Submit a managed job to SkyPilot and record it in the platform DB."""
    import sky
    from sky.jobs.server import core as jobs_core

    wtype = CloudWorkloadType(workload_type)

    # Build task from YAML
    task_config = yaml.safe_load(task_yaml)
    task = sky.Task.from_yaml_config(task_config)

    if accelerators:
        task.set_resources(
            sky.Resources(accelerators=accelerators, use_spot=use_spot)
        )

    # Launch via SkyPilot (blocking — run in thread)
    result = await asyncio.to_thread(
        jobs_core.launch, task, name=name, stream_logs=False
    )
    sky_job_id = None
    if result is not None:
        job_ids, _handle = result
        if isinstance(job_ids, list) and job_ids:
            sky_job_id = job_ids[0]
        elif isinstance(job_ids, int):
            sky_job_id = job_ids

    job = await repo.create_managed_job(
        session,
        name=name,
        project_id=project_id,
        workload_type=wtype,
        requested_by_id=requested_by_id,
        task_yaml=task_yaml,
        skypilot_job_id=sky_job_id,
        status="queued" if sky_job_id is not None else "failed",
        accelerators=accelerators,
        cloud=cloud,
        use_spot=use_spot,
    )
    return job


async def list_jobs(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    type_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
):
    """List managed jobs merged from SkyPilot state + platform DB."""
    wtype = CloudWorkloadType(type_filter) if type_filter else None
    db_jobs = await repo.list_managed_jobs(
        session,
        project_id=project_id,
        type_filter=wtype,
        status=status_filter,
        limit=limit,
    )

    # Try to sync status from SkyPilot for jobs that are still active
    try:
        sky_statuses = await _get_managed_job_statuses()
    except Exception:
        logger.debug("Could not fetch SkyPilot job statuses", exc_info=True)
        sky_statuses = {}

    results = []
    status_counts: dict[str, int] = {}
    for job in db_jobs:
        # Sync status from SkyPilot if available
        if job.skypilot_job_id is not None and job.skypilot_job_id in sky_statuses:
            new_status = _map_status(sky_statuses[job.skypilot_job_id])
            if new_status != job.status:
                started = datetime.utcnow() if new_status == "running" and job.started_at is None else None
                completed = datetime.utcnow() if new_status in ("completed", "failed", "cancelled") else None
                await repo.update_managed_job_status(
                    session, job.id, new_status,
                    started_at=started, completed_at=completed,
                )
                job.status = new_status

        status_counts[job.status] = status_counts.get(job.status, 0) + 1
        results.append(job)

    return {
        "jobs": results,
        "total": len(results),
        "status_counts": status_counts,
    }


async def cancel_job(session: AsyncSession, job_id: str):
    """Cancel a managed job."""
    from sky import core as sky_core

    job = await repo.get_managed_job(session, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    if job.skypilot_job_id is not None:
        try:
            await asyncio.to_thread(
                sky_core.cancel,
                cluster_name=f"sky-jobs-{job.skypilot_job_id}",
            )
        except Exception:
            logger.warning(
                f"SkyPilot cancel failed for job {job.skypilot_job_id}",
                exc_info=True,
            )

    await repo.update_managed_job_status(
        session, job.id, "cancelled", completed_at=datetime.utcnow()
    )


# ── Cluster / Cloud status ──────────────────────────────────────────


async def get_cluster_status() -> list:
    """Return SkyPilot cluster status (all clusters)."""
    from sky import core as sky_core

    return await asyncio.to_thread(sky_core.status)


async def get_cost_report(days: int = 30) -> list:
    """Return SkyPilot cost report."""
    from sky import core as sky_core

    return await asyncio.to_thread(sky_core.cost_report, days=days)


async def terminate_cluster(cluster_name: str):
    """Tear down a SkyPilot cluster."""
    from sky import core as sky_core

    await asyncio.to_thread(sky_core.down, cluster_name=cluster_name)


# ── Helpers ──────────────────────────────────────────────────────────


async def _get_managed_job_statuses() -> dict[int, str]:
    """Fetch current status for all managed jobs from SkyPilot's state DB."""
    try:
        from sky.jobs import state as job_state

        records = await asyncio.to_thread(
            job_state.get_managed_jobs,
        )
        return {r["job_id"]: str(r["status"]) for r in records}
    except Exception:
        return {}
