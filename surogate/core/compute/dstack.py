"""Compute service — wraps dstack internal services.

dstack is used as an embedded library.  Its APScheduler background tasks
drive run state transitions.  This module provides the bridge between
Surogate's own DB/API and dstack's run lifecycle.

All dstack DB access goes through ``dstack_session_ctx()`` — a separate
async session scoped to dstack's own SQLite database.
"""

import asyncio
import re
from datetime import datetime
from typing import Optional

import sqlalchemy as sa
import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute import (
    ensure_dstack_project,
    get_dstack_admin,
    get_dstack_project,
)
from surogate.core.db.models.compute import CloudWorkloadType, ServingService
from surogate.core.db.models.platform import Project
from surogate.core.db.repository import compute as repo
from surogate.utils.logger import get_logger

logger = get_logger()


# ── Status mapping ───────────────────────────────────────────────────


def map_status(dstack_status) -> str:
    """Map a dstack RunStatus to a Surogate status string."""
    from dstack._internal.core.models.runs import RunStatus

    _STATUS_MAP = {
        RunStatus.PENDING: "queued",
        RunStatus.SUBMITTED: "queued",
        RunStatus.PROVISIONING: "provisioning",
        RunStatus.RUNNING: "running",
        RunStatus.TERMINATING: "cancelling",
        RunStatus.TERMINATED: "cancelled",
        RunStatus.FAILED: "failed",
        RunStatus.DONE: "completed",
    }
    if isinstance(dstack_status, str):
        try:
            dstack_status = RunStatus(dstack_status)
        except ValueError:
            return "unknown"
    return _STATUS_MAP.get(dstack_status, "unknown")


# ── Helpers ──────────────────────────────────────────────────────────


async def _resolve_dstack_project(session: AsyncSession, project_id: str):
    """Look up the surogate project, lazily create the dstack project, return dstack ProjectModel."""
    result = await session.execute(
        sa.select(Project.namespace, Project.dstack_project_name)
        .where(Project.id == project_id)
    )
    row = result.one_or_none()
    if row is None:
        raise ValueError(f"Surogate project {project_id} not found")

    namespace, dstack_name = row
    if dstack_name:
        project = await get_dstack_project(dstack_name)
        if project is not None:
            return project

    # Lazily create
    project = await ensure_dstack_project(namespace)

    # Persist the mapping back to surogate DB
    await session.execute(
        sa.update(Project)
        .where(Project.id == project_id)
        .values(dstack_project_name=project.name)
    )
    await session.commit()
    return project


def _dstack_name(base: str, prefix: str = "") -> str:
    """Sanitize a name to match dstack's ``^[a-z][a-z0-9-]{1,40}$`` (max 41 chars)."""
    import hashlib
    slug = re.sub(r"[^a-z0-9-]", "-", base.lower()).strip("-")
    if prefix:
        slug = f"{prefix}-{slug}"
    if len(slug) <= 41:
        return slug
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    return f"{slug[:32]}-{h}"


def _parse_accelerators(spec: Optional[str]):
    """Parse ``"H100:2"`` into ``("H100", 2)``."""
    if not spec:
        return None, 1
    parts = spec.split(":", 1)
    name = parts[0]
    count = int(parts[1]) if len(parts) > 1 else 1
    return name, count


async def _create_fleet_for_run(
    dstack_project,
    *,
    name: str,
    gpu_spec=None,
    use_spot: bool = False,
) -> None:
    """Create a fleet tailored to a run's resource requirements.

    Each run gets its own fleet so dstack provisions instances matching
    the exact GPU / spot / resource needs.
    """
    from dstack._internal.core.models.fleets import FleetConfiguration, FleetSpec
    from dstack._internal.core.models.profiles import Profile, SpotPolicy
    from dstack._internal.core.models.resources import ResourcesSpec
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.fleets import (
        create_fleet,
        get_project_fleet_model_by_name,
    )
    from dstack._internal.server.services.pipelines import _noop_pipeline_hinter

    admin = get_dstack_admin()

    # Delete any existing fleet with the same name (from a previous run).
    # We force-mark it deleted since delete_fleets only initiates termination
    # and the fleet may get stuck in TERMINATING.
    async with dstack_session_ctx() as dstack_session:
        existing = await get_project_fleet_model_by_name(
            session=dstack_session, project=dstack_project, name=name,
        )
        if existing is not None:
            from dstack._internal.core.models.fleets import FleetStatus
            from dstack._internal.server.models import FleetModel
            await dstack_session.execute(
                sa.update(FleetModel)
                .where(FleetModel.id == existing.id)
                .values(status=FleetStatus.TERMINATED, deleted=True)
            )
            await dstack_session.commit()

    from dstack._internal.core.models.fleets import FleetNodesSpec

    fleet_config = FleetConfiguration(
        name=name,
        nodes=FleetNodesSpec(min=0, max=1),
        spot_policy=SpotPolicy.SPOT if use_spot else SpotPolicy.ONDEMAND,
    )
    if gpu_spec:
        fleet_config.resources = ResourcesSpec(gpu=gpu_spec)

    spec = FleetSpec(
        configuration=fleet_config,
        profile=Profile(name="default"),
    )

    async with dstack_session_ctx() as dstack_session:
        await create_fleet(
            session=dstack_session,
            project=dstack_project,
            user=admin,
            spec=spec,
            pipeline_hinter=_noop_pipeline_hinter,
        )
    logger.info("Created fleet '%s' for project '%s'", name, dstack_project.name)


# ── Task (Managed Job) Operations ───────────────────────────────────


async def submit_task(
    session: AsyncSession,
    *,
    name: str,
    project_id: str,
    workload_type: str,
    requested_by_id: str,
    image: str,
    commands: list[str],
    accelerators: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    use_spot: bool = False,
):
    """Submit a task run via dstack and record it in the surogate DB."""
    from dstack._internal.core.models.configurations import TaskConfiguration
    from dstack._internal.core.models.profiles import Profile, SpotPolicy
    from dstack._internal.core.models.resources import GPUSpec, ResourcesSpec
    from dstack._internal.core.models.runs import RunSpec
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.runs import submit_run

    wtype = CloudWorkloadType(workload_type)
    dstack_project = await _resolve_dstack_project(session, project_id)
    admin = get_dstack_admin()

    # Build configuration
    gpu_name, gpu_count = _parse_accelerators(accelerators)
    gpu_spec = None
    if gpu_name:
        gpu_spec = GPUSpec(name=[gpu_name], count=gpu_count)

    task_config = TaskConfiguration(
        type="task",
        name=_dstack_name(name),
        image=image,
        commands=commands,
        env=env or {},
        spot_policy=SpotPolicy.SPOT if use_spot else SpotPolicy.ONDEMAND,
    )
    if gpu_spec:
        task_config.resources = ResourcesSpec(gpu=gpu_spec)

    run_spec = RunSpec(
        configuration=task_config,
        profile=Profile(name="default"),
    )

    # Create a fleet matching this run's requirements
    fleet_name = _dstack_name(name, prefix="f")
    await _create_fleet_for_run(
        dstack_project, name=fleet_name, gpu_spec=gpu_spec, use_spot=use_spot,
    )

    # Submit to dstack
    async with dstack_session_ctx() as dstack_session:
        run = await submit_run(
            session=dstack_session,
            user=admin,
            project=dstack_project,
            run_spec=run_spec,
        )

    run_name = run.run_spec.run_name

    # Record in surogate DB
    job = await repo.create_managed_job(
        session,
        name=name,
        project_id=project_id,
        workload_type=wtype,
        requested_by_id=requested_by_id,
        task_yaml=yaml.dump({"image": image, "commands": commands, "env": env or {}}),
        dstack_run_name=run_name,
        dstack_project_name=dstack_project.name,
        status="queued",
        accelerators=accelerators,
        use_spot=use_spot,
    )
    return job


async def list_tasks(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    type_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
):
    """List managed tasks, syncing status from dstack for active ones."""
    wtype = CloudWorkloadType(type_filter) if type_filter else None
    db_jobs = await repo.list_managed_jobs(
        session,
        project_id=project_id,
        type_filter=wtype,
        status=status_filter,
        limit=limit,
    )

    # Bulk-sync active jobs from dstack
    try:
        # Group by dstack project
        project_names = {
            j.dstack_project_name
            for j in db_jobs
            if j.dstack_run_name and j.dstack_project_name
        }
        all_statuses: dict[str, str] = {}
        for pname in project_names:
            statuses = await get_active_run_statuses(pname)
            all_statuses.update(statuses)
    except Exception:
        logger.debug("Could not fetch dstack run statuses", exc_info=True)
        all_statuses = {}

    results = []
    status_counts: dict[str, int] = {}
    now = datetime.utcnow()
    for job in db_jobs:
        if job.dstack_run_name and job.dstack_run_name in all_statuses:
            new_status = all_statuses[job.dstack_run_name]
            if new_status != job.status:
                started = now if new_status == "running" and job.started_at is None else None
                completed = now if new_status in ("completed", "failed", "cancelled") else None
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


async def cancel_task(session: AsyncSession, job_id: str):
    """Cancel a managed task."""
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.runs import stop_runs

    job = await repo.get_managed_job(session, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    if job.dstack_run_name and job.dstack_project_name:
        dstack_project = await get_dstack_project(job.dstack_project_name)
        if dstack_project is not None:
            admin = get_dstack_admin()
            try:
                async with dstack_session_ctx() as dstack_session:
                    await stop_runs(
                        session=dstack_session,
                        user=admin,
                        project=dstack_project,
                        runs_names=[job.dstack_run_name],
                        abort=False,
                    )
            except Exception:
                logger.warning(
                    "dstack stop_runs failed for job %s", job.dstack_run_name,
                    exc_info=True,
                )

    await repo.update_managed_job_status(
        session, job.id, "cancelled", completed_at=datetime.utcnow()
    )


# ── Service Operations ──────────────────────────────────────────────


async def launch_service(
    session: AsyncSession,
    svc: ServingService,
    *,
    image: str,
    commands: list[str],
    port: int = 8080,
    env: Optional[dict[str, str]] = None,
    readiness_probe: str = "/health",
):
    """Launch a dstack service run and update the surogate DB record."""
    from dstack._internal.core.models.configurations import (
        ProbeConfig,
        ServiceConfiguration,
    )
    from dstack._internal.core.models.profiles import Profile, SpotPolicy
    from dstack._internal.core.models.resources import GPUSpec, ResourcesSpec
    from dstack._internal.core.models.runs import RunSpec
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.runs import submit_run

    dstack_project = await _resolve_dstack_project(session, svc.project_id)
    admin = get_dstack_admin()

    gpu_name, gpu_count = _parse_accelerators(svc.accelerators)
    gpu_spec = None
    if gpu_name:
        gpu_spec = GPUSpec(name=[gpu_name], count=gpu_count)

    service_config = ServiceConfiguration(
        type="service",
        name=_dstack_name(svc.name),
        image=image,
        port=port,
        commands=commands,
        replicas=svc.replicas,
        gateway=False,
        auth=False,
        env=env or {},
        spot_policy=SpotPolicy.SPOT if svc.use_spot else SpotPolicy.ONDEMAND,
        probes=[ProbeConfig(type="http", url=readiness_probe)],
    )
    if gpu_spec:
        service_config.resources = ResourcesSpec(gpu=gpu_spec)

    run_spec = RunSpec(
        configuration=service_config,
        profile=Profile(name="default"),
    )

    status = "submitted"
    run_name = None
    endpoint = None

    try:
        # Create a fleet matching this service's requirements
        fleet_name = _dstack_name(svc.name, prefix="f")
        await _create_fleet_for_run(
            dstack_project, name=fleet_name, gpu_spec=gpu_spec, use_spot=svc.use_spot,
        )

        async with dstack_session_ctx() as dstack_session:
            run = await submit_run(
                session=dstack_session,
                user=admin,
                project=dstack_project,
                run_spec=run_spec,
            )
        run_name = run.run_spec.run_name
        endpoint = f"/proxy/services/{dstack_project.name}/{run_name}"
    except Exception as exc:
        logger.warning(
            "dstack service launch failed for %s: %s", svc.name, exc,
            exc_info=True,
        )
        status = "failed"

    await repo.update_serving_service(
        session, svc.id,
        task_yaml=yaml.dump({"image": image, "commands": commands, "port": port, "env": env or {}}),
        status=status,
        endpoint=endpoint,
        dstack_run_name=run_name,
        dstack_project_name=dstack_project.name,
    )
    return await repo.get_serving_service(session, svc.id)


async def terminate_service(session: AsyncSession, service_id: str):
    """Stop a dstack service run and unregister from the proxy."""
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.runs import stop_runs

    svc = await repo.get_serving_service(session, service_id)
    if svc is None:
        raise ValueError(f"Serving service {service_id} not found")

    if svc.dstack_run_name and svc.dstack_project_name:
        dstack_project = await get_dstack_project(svc.dstack_project_name)
        if dstack_project is not None:
            admin = get_dstack_admin()
            try:
                async with dstack_session_ctx() as dstack_session:
                    await stop_runs(
                        session=dstack_session,
                        user=admin,
                        project=dstack_project,
                        runs_names=[svc.dstack_run_name],
                        abort=True,
                    )
            except Exception as ex:
                logger.warning(
                    "dstack stop_runs failed for service %s: %s",
                    svc.name, ex, exc_info=True,
                )

    await repo.update_serving_service(
        session, svc.id,
        status="stopped",
        terminated_at=datetime.utcnow(),
    )


# ── Logs ────────────────────────────────────────────────────────────


async def get_run_logs(
    run_name: str,
    project_name: str,
    *,
    tail: int = 200,
) -> list[str]:
    """Fetch logs for a dstack run."""
    from base64 import b64decode
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.logs import poll_logs_async
    from dstack._internal.server.services.projects import get_project_model_by_name
    from dstack._internal.server.schemas.logs import PollLogsRequest

    try:
        async with dstack_session_ctx() as dstack_session:
            project = await get_project_model_by_name(
                session=dstack_session, project_name=project_name,
            )
            if project is None:
                return []

            from dstack._internal.server.services.runs import get_run
            run = await get_run(
                session=dstack_session, project=project, run_name=run_name,
            )
            if run is None or not run.jobs:
                return []

            job_sub = run.latest_job_submission or (
                run.jobs[0].job_submissions[-1]
                if run.jobs and run.jobs[0].job_submissions
                else None
            )
            if job_sub is None:
                return []

        # poll_logs_async uses run_async internally, call outside the session
        request = PollLogsRequest(
            run_name=run_name,
            job_submission_id=job_sub.id,
            limit=tail,
            diagnose=False,
            descending=True,
        )
        result = await poll_logs_async(project=project, request=request)

        lines = []
        for entry in result.logs:
            # poll_logs_async base64-encodes messages for API compat
            try:
                msg = b64decode(entry.message).decode("utf-8", errors="replace")
            except Exception:
                msg = entry.message
            # Each message may contain multiple lines
            lines.extend(msg.rstrip("\n").split("\n"))

        # descending=True returns newest first, reverse for chronological order
        lines.reverse()
        return lines[-tail:]
    except Exception:
        logger.debug("Could not fetch logs for run %s", run_name, exc_info=True)
        return []


# ── Instance Operations ─────────────────────────────────────────────


async def list_instances() -> list[dict]:
    """List all active dstack instances across all projects."""
    from dstack._internal.core.models.instances import InstanceStatus
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.models import InstanceModel

    try:
        async with dstack_session_ctx() as dstack_session:
            from sqlalchemy import select
            stmt = select(InstanceModel).where(
                InstanceModel.status.notin_([
                    InstanceStatus.TERMINATED.value,
                    InstanceStatus.TERMINATING.value,
                ])
            )
            result = await dstack_session.execute(stmt)
            instances = result.scalars().all()

            out = []
            for inst in instances:
                out.append({
                    "id": str(inst.id),
                    "name": inst.name,
                    "status": inst.status.value if hasattr(inst.status, "value") else str(inst.status),
                    "project_id": str(inst.project_id),
                })
            return out
    except Exception:
        logger.debug("Could not list dstack instances", exc_info=True)
        return []


async def terminate_instance(instance_id: str, project_name: str):
    """Terminate a dstack instance."""
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.projects import get_project_model_by_name

    async with dstack_session_ctx() as dstack_session:
        project = await get_project_model_by_name(
            session=dstack_session, project_name=project_name,
        )
        if project is None:
            raise ValueError(f"dstack project {project_name} not found")

        # Mark instance for termination
        from dstack._internal.server.models import InstanceModel
        from dstack._internal.core.models.instances import InstanceStatus
        await dstack_session.execute(
            sa.update(InstanceModel)
            .where(InstanceModel.id == instance_id)
            .values(status=InstanceStatus.TERMINATING)
        )
        await dstack_session.commit()


# ── Replica URL Discovery (for proxy) ──────────────────────────────


async def get_service_replica_urls(
    run_name: str, project_name: str
) -> list[str]:
    """Get replica URLs from running jobs of a dstack service run.

    Inspects ``job_provisioning_data`` on each RUNNING job to extract
    the instance hostname and mapped service port.
    """
    from dstack._internal.core.models.runs import JobStatus
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.projects import get_project_model_by_name
    from dstack._internal.server.services.runs import get_run

    try:
        async with dstack_session_ctx() as dstack_session:
            project = await get_project_model_by_name(
                session=dstack_session, project_name=project_name,
            )
            if project is None:
                return []

            run = await get_run(
                session=dstack_session, project=project, run_name=run_name,
            )
            if run is None:
                return []

            # Get the container port from the service configuration
            container_port = 8080
            cfg = run.run_spec.configuration
            if hasattr(cfg, "port") and cfg.port:
                port_spec = cfg.port
                if hasattr(port_spec, "container_port"):
                    container_port = port_spec.container_port
                elif isinstance(port_spec, int):
                    container_port = port_spec

            urls = []
            for job in run.jobs:
                latest = job.job_submissions[-1] if job.job_submissions else None
                if latest is None or latest.status != JobStatus.RUNNING:
                    continue
                prov = latest.job_provisioning_data
                if prov is None or not prov.hostname:
                    continue
                urls.append(f"http://{prov.hostname}:{container_port}")
            return urls
    except Exception:
        logger.debug(
            "Could not discover replica URLs for run %s", run_name,
            exc_info=True,
        )
        return []


# ── Monitor Helpers ─────────────────────────────────────────────────


async def get_active_run_statuses(project_name: str) -> dict[str, str]:
    """Bulk-fetch mapped statuses for all non-terminal runs in a dstack project.

    Returns ``{run_name: surogate_status_string}``.
    """
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.projects import get_project_model_by_name
    from dstack._internal.server.services.runs import list_user_runs

    try:
        async with dstack_session_ctx() as dstack_session:
            project = await get_project_model_by_name(
                session=dstack_session, project_name=project_name,
            )
            if project is None:
                return {}

            admin = get_dstack_admin()
            runs = await list_user_runs(
                session=dstack_session,
                user=admin,
                project_name=project_name,
                repo_id=None,
                username=None,
                only_active=True,
                include_jobs=False,
                job_submissions_limit=None,
                prev_submitted_at=None,
                prev_run_id=None,
                limit=500,
                ascending=False,
            )
            return {
                r.run_spec.run_name: map_status(r.status)
                for r in runs
                if r.run_spec.run_name
            }
    except Exception:
        logger.debug(
            "Could not fetch active run statuses for project %s",
            project_name, exc_info=True,
        )
        return {}
