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

from surogate.core.db.models.compute import CloudWorkloadType, ServingService
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


def _map_status(sky_status) -> str:
    key = sky_status.name if hasattr(sky_status, "name") else str(sky_status)
    return _SKY_STATUS_MAP.get(key.upper(), "unknown")


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

async def launch_serving_service(
    session: AsyncSession,
    svc: ServingService,
    task_yaml: str,
):
    """Launch a SkyPilot serving service and update the existing DB record."""
    import sky
    from sky.serve.server import core as serve_core

    task_config = yaml.safe_load(task_yaml)
    task = sky.Task.from_yaml_config(task_config)

    endpoint = None
    controller_port = None
    status = "controller_init"
    try:
        from surogate.core.compute.skypilot.patcher import _active_project_id

        _active_project_id.set(svc.project_id)
        await asyncio.to_thread(
            serve_core.up, task, service_name=svc.name
        )
        # Register with our reverse proxy (replaces SkyPilot's LB process)
        from sky.serve import serve_state
        from surogate.server.routes.proxy import register_service
        controller_port = serve_state.get_service_controller_port(svc.name)
        register_service(svc.name, controller_port)
        endpoint = f"/api/serve/{svc.name}"
    except Exception as exc:
        logger.warning(
            f"SkyPilot serve launch failed for {svc.name}: {exc}",
            exc_info=True,
        )
        status = "failed"

    await repo.update_serving_service(
        session, svc.id,
        task_yaml=task_yaml,
        status=status,
        endpoint=endpoint,
        controller_port=controller_port,
    )
    return await repo.get_serving_service(session, svc.id)


async def get_serving_service_status(service_name: str) -> Optional[dict]:
    """Get detailed status for a single serving service from SkyPilot."""
    from sky.serve.server import core as serve_core

    try:
        result = await asyncio.to_thread(serve_core.status, service_names=service_name)
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
    from sky.serve.server import core as serve_core
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
        serve_core.update, task, service_name=svc.name, mode=update_mode
    )

    await repo.update_serving_service(
        session, svc.id, task_yaml=task_yaml, update_mode=mode
    )
    return svc


async def terminate_serving_service(session: AsyncSession, service_id: str, purge: bool = False):
    """Tear down a serving service."""
    from sky.serve.server import core as serve_core

    svc = await repo.get_serving_service(session, service_id)
    if svc is None:
        raise ValueError(f"Serving service {service_id} not found")

    from surogate.server.routes.proxy import unregister_service
    unregister_service(svc.name)

    try:
        await asyncio.to_thread(serve_core.down, service_names=svc.name, purge=purge)
        status = "stopped"
    except Exception as ex:
        logger.warning(
            f"SkyPilot serve down failed for {svc.name}: {ex}", exc_info=True
        )
        status = "failed_cleanup"

    await repo.update_serving_service(
        session, svc.id,
        status=status,
        terminated_at=datetime.utcnow(),
    )


async def terminate_serving_service_replica(
    service_name: str, replica_id: int, purge: bool = False
):
    """Terminate a specific replica of a serving service."""
    from sky.serve.server import core as serve_core

    await asyncio.to_thread(
        serve_core.terminate_replica,
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


async def tail_serving_logs(
    service_name: str,
    *,
    target: str = "controller",
    replica_id: Optional[int] = None,
    tail: int = 200,
) -> list[str]:
    """Read the last N lines of logs for a serving service.

    *target* is one of ``"controller"``, ``"load_balancer"``, or
    ``"replica"`` (requires *replica_id*).
    """
    from sky.serve import serve_utils
    from sky import backends
    from sky.backends import backend_utils
    from sky.utils import controller_utils

    component_map = {
        "controller": serve_utils.ServiceComponent.CONTROLLER,
        "load_balancer": serve_utils.ServiceComponent.LOAD_BALANCER,
        "replica": serve_utils.ServiceComponent.REPLICA,
    }
    component = component_map.get(target)
    if component is None:
        raise ValueError(f"Unknown log target: {target!r}")

    if component == serve_utils.ServiceComponent.REPLICA:
        if replica_id is None:
            raise ValueError("`replica_id` required for replica logs")
        code = serve_utils.ServeCodeGen.stream_replica_logs(
            service_name, replica_id, follow=False, tail=tail, pool=False,
        )
    else:
        code = serve_utils.ServeCodeGen.stream_serve_process_logs(
            service_name,
            stream_controller=(component == serve_utils.ServiceComponent.CONTROLLER),
            follow=False,
            tail=tail,
            pool=False,
        )

    def _fetch() -> str:
        controller_type = controller_utils.get_controller_for_pool(False)
        handle = backend_utils.is_controller_accessible(
            controller=controller_type,
            stopped_message=controller_type.value.default_hint_if_non_existent,
        )
        backend = backend_utils.get_backend_from_handle(handle)
        assert isinstance(backend, backends.CloudVmRayBackend)
        _, stdout, _ = backend.run_on_head(
            handle, code,
            stream_logs=False,
            require_outputs=True,
            separate_stderr=True,
        )
        return stdout

    try:
        stdout = await asyncio.to_thread(_fetch)
    except Exception:
        logger.debug(
            "Could not fetch logs for serving service %s", service_name,
            exc_info=True,
        )
        return []

    lines = stdout.splitlines()
    return lines[-tail:] if len(lines) > tail else lines


async def _get_serve_statuses() -> dict[str, dict]:
    """Fetch current status for all serving services from SkyPilot."""
    try:
        from sky.serve.server import core as serve_core

        records = await asyncio.to_thread(serve_core.status, service_names=None)
        if not records:
            return {}
        out: dict[str, dict] = {}
        for r in records:
            # SkyPilot returns ServiceStatus enums; convert to plain strings
            # so they can be bound as SQL parameters.
            if hasattr(r.get("status"), "value"):
                r["status"] = r["status"].value.lower()
            out[r["name"]] = r
        return out
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
