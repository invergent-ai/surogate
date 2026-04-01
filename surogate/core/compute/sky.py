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


# ── SkyPilot Serve ──────────────────────────────────────────────────

_SKY_SERVING_SERVICE_STATUS_MAP = {
    "CONTROLLER_INIT": "controller_init",
    "REPLICA_INIT": "replica_init",
    "CONTROLLER_FAILED": "controller_failed",
    "READY": "ready",
    "SHUTTING_DOWN": "shutting_down",
    "FAILED": "failed",
    "FAILED_CLEANUP": "failed_cleanup",
    "NO_REPLICA": "no_replica",
}


def _map_serving_service_status(sky_status: str) -> str:
    return _SKY_SERVING_SERVICE_STATUS_MAP.get(str(sky_status).upper(), "unknown")


async def launch_serving_service(
    session: AsyncSession,
    *,
    task_yaml: str,
    service_name: str,
    project_id: str,
    requested_by_id: str,
    accelerators: Optional[str] = None,
    cloud: Optional[str] = None,
    use_spot: bool = False,
    min_replicas: int = 1,
    max_replicas: Optional[int] = None,
    readiness_path: Optional[str] = None,
    load_balancing_policy: Optional[str] = None,
):
    """Launch a SkyPilot serving service and record it in the platform DB."""
    import sky
    from sky import serve as sky_serve

    task_config = yaml.safe_load(task_yaml)
    task = sky.Task.from_yaml_config(task_config)

    if accelerators:
        task.set_resources(
            sky.Resources(accelerators=accelerators, use_spot=use_spot)
        )

    # Build service section if not already in the YAML
    service_config: dict = task_config.get("service", {})
    if readiness_path:
        service_config["readiness_probe"] = readiness_path
    if min_replicas:
        service_config.setdefault("replica_policy", {})["min_replicas"] = min_replicas
    if max_replicas:
        service_config.setdefault("replica_policy", {})["max_replicas"] = max_replicas
    if load_balancing_policy:
        service_config["load_balancing_policy"] = load_balancing_policy

    if service_config and not task_config.get("service"):
        task_config["service"] = service_config
        task = sky.Task.from_yaml_config(task_config)

    endpoint = None
    status = "controller_init"
    try:
        request_id = await asyncio.to_thread(
            sky_serve.up, task, service_name=service_name
        )
        # sky.serve.up returns a RequestId; resolve it to get (name, endpoint)
        if request_id is not None:
            result = await asyncio.to_thread(request_id.get)
            if result and len(result) >= 2:
                endpoint = result[1]
    except Exception:
        logger.warning(
            f"SkyPilot serve launch failed for {service_name}", exc_info=True
        )
        status = "failed"

    svc = await repo.create_serving_service(
        session,
        name=service_name,
        project_id=project_id,
        requested_by_id=requested_by_id,
        task_yaml=task_yaml,
        status=status,
        endpoint=endpoint,
        accelerators=accelerators,
        cloud=cloud,
        use_spot=use_spot,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        readiness_path=readiness_path,
        load_balancing_policy=load_balancing_policy,
    )
    return svc


async def list_serving_services(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
):
    """List serving services merged from SkyPilot state + platform DB."""
    db_services = await repo.list_serving_services(
        session,
        project_id=project_id,
        status=status_filter,
        limit=limit,
    )

    # Sync status from SkyPilot
    try:
        sky_statuses = await _get_serve_statuses()
    except Exception:
        logger.debug("Could not fetch SkyPilot serve statuses", exc_info=True)
        sky_statuses = {}

    results = []
    status_counts: dict[str, int] = {}
    for svc in db_services:
        if svc.name in sky_statuses:
            info = sky_statuses[svc.name]
            new_status = _map_serving_service_status(info.get("status", ""))
            updates: dict = {}
            if new_status != svc.status:
                updates["status"] = new_status
                if new_status == "ready" and svc.started_at is None:
                    updates["started_at"] = datetime.utcnow()
                if new_status in ("failed", "failed_cleanup", "controller_failed"):
                    updates["terminated_at"] = datetime.utcnow()
            ep = info.get("endpoint")
            if ep and ep != svc.endpoint:
                updates["endpoint"] = ep
            if updates:
                await repo.update_serving_service(session, svc.id, **updates)
                for k, v in updates.items():
                    setattr(svc, k, v)

        status_counts[svc.status] = status_counts.get(svc.status, 0) + 1
        results.append(svc)

    return {
        "services": results,
        "total": len(results),
        "status_counts": status_counts,
    }


async def get_serving_service_status(service_name: str) -> Optional[dict]:
    """Get detailed status for a single serving service from SkyPilot."""
    from sky import serve as sky_serve

    try:
        request_id = await asyncio.to_thread(sky_serve.status, service_name)
        result = await asyncio.to_thread(request_id.get)
        if result and len(result) > 0:
            return result[0]
    except Exception:
        logger.debug(f"Could not fetch status for serving service {service_name}", exc_info=True)
    return None


async def update_serving_service(
    session: AsyncSession,
    service_id: str,
    *,
    task_yaml: str,
    mode: str = "rolling",
):
    """Update a serving service with a new task configuration."""
    import sky
    from sky import serve as sky_serve
    from sky.serve import serve_utils

    svc = await repo.get_serving_service(session, service_id)
    if svc is None:
        raise ValueError(f"Service {service_id} not found")

    task_config = yaml.safe_load(task_yaml)
    task = sky.Task.from_yaml_config(task_config)

    update_mode = (
        serve_utils.UpdateMode.BLUE_GREEN
        if mode == "blue_green"
        else serve_utils.UpdateMode.ROLLING
    )

    await asyncio.to_thread(
        sky_serve.update, task, service_name=svc.name, mode=update_mode
    )

    await repo.update_serving_service(
        session, svc.id, task_yaml=task_yaml, update_mode=mode
    )
    return svc


async def terminate_serving_service(session: AsyncSession, service_id: str, purge: bool = False):
    """Tear down a serving service."""
    from sky import serve as sky_serve

    svc = await repo.get_serving_service(session, service_id)
    if svc is None:
        raise ValueError(f"Serving service {service_id} not found")

    try:
        await asyncio.to_thread(sky_serve.down, svc.name, purge=purge)
    except Exception:
        logger.warning(
            f"SkyPilot serve down failed for {svc.name}", exc_info=True
        )

    await repo.update_serving_service(
        session, svc.id,
        status="shutting_down",
        terminated_at=datetime.utcnow(),
    )


async def terminate_serving_service_replica(
    service_name: str, replica_id: int, purge: bool = False
):
    """Terminate a specific replica of a serving service."""
    from sky import serve as sky_serve

    await asyncio.to_thread(
        sky_serve.terminate_replica,
        service_name=service_name,
        replica_id=replica_id,
        purge=purge,
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


async def _get_serve_statuses() -> dict[str, dict]:
    """Fetch current status for all serving services from SkyPilot."""
    try:
        from sky import serve as sky_serve

        request_id = await asyncio.to_thread(sky_serve.status, None)
        records = await asyncio.to_thread(request_id.get)
        return {r["name"]: r for r in records} if records else {}
    except Exception:
        return {}


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
