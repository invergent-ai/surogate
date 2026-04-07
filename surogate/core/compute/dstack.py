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


async def delete_backend(session: AsyncSession, project_id: str, backend_type: str) -> None:
    """Terminate all instances on a backend, then remove it from the project."""
    from dstack._internal.core.models.backends.base import BackendType
    from dstack._internal.core.models.instances import InstanceStatus
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.models import InstanceModel
    from dstack._internal.server.services.backends import delete_backends

    dstack_project = await _resolve_dstack_project(session, project_id)
    bt = BackendType(backend_type)

    # Terminate all active instances on this backend
    async with dstack_session_ctx() as dstack_session:
        await dstack_session.execute(
            sa.update(InstanceModel)
            .where(
                InstanceModel.project_id == dstack_project.id,
                InstanceModel.backend == bt,
                InstanceModel.deleted == False,
                InstanceModel.status.notin_([
                    InstanceStatus.TERMINATED,
                    InstanceStatus.TERMINATING,
                ]),
            )
            .values(status=InstanceStatus.TERMINATING)
        )
        await dstack_session.commit()

    # Remove the backend and invalidate the cache
    async with dstack_session_ctx() as dstack_session:
        await delete_backends(dstack_session, dstack_project, [bt])

    from surogate.core.compute import _invalidate_backends_cache
    _invalidate_backends_cache(dstack_project)


async def list_backends(session: AsyncSession, project_id: str) -> list[dict]:
    """Return the registered dstack backends for a surogate project with instance stats."""
    from dstack._internal.core.models.instances import InstanceStatus
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.models import InstanceModel

    dstack_project = await _resolve_dstack_project(session, project_id)

    # Query active instance counts and cost per backend type for this project
    instance_stats: dict[str, dict] = {}
    try:
        async with dstack_session_ctx() as dstack_session:
            stmt = (
                sa.select(
                    InstanceModel.backend,
                    sa.func.count().label("count"),
                    sa.func.coalesce(sa.func.sum(InstanceModel.price), 0.0).label("hourly_cost"),
                )
                .where(
                    InstanceModel.project_id == dstack_project.id,
                    InstanceModel.deleted == False,
                    InstanceModel.status.notin_([
                        InstanceStatus.TERMINATED,
                        InstanceStatus.TERMINATING,
                    ]),
                )
                .group_by(InstanceModel.backend)
            )
            for row in (await dstack_session.execute(stmt)).all():
                backend_type = row.backend.value if hasattr(row.backend, "value") else str(row.backend)
                instance_stats[backend_type] = {
                    "active_instances": row.count,
                    "hourly_cost": round(float(row.hourly_cost), 2),
                }
    except Exception:
        logger.debug("Could not fetch backend instance stats", exc_info=True)

    return [
        {
            "type": b.type.value,
            "id": str(b.id),
            **(instance_stats.get(b.type.value, {"active_instances": 0, "hourly_cost": 0.0})),
        }
        for b in dstack_project.backends
    ]


async def _fetch_offers(backends) -> list[dict]:
    """Fetch offers from a list of dstack Backend objects and return serialised dicts."""
    from dstack._internal.core.models.resources import ResourcesSpec
    from dstack._internal.core.models.runs import Requirements
    from dstack._internal.server.services.backends import get_backend_offers

    requirements = Requirements(resources=ResourcesSpec())
    offers_iter = await get_backend_offers(backends, requirements)

    seen = set()
    results = []
    for backend, offer in offers_iter:
        key = (backend.TYPE.value, offer.instance.name, offer.region)
        if key in seen:
            continue
        seen.add(key)

        gpus = offer.instance.resources.gpus
        results.append({
            "backend": backend.TYPE.value,
            "instance": offer.instance.name,
            "region": offer.region,
            "price": round(offer.price, 4),
            "cpus": offer.instance.resources.cpus,
            "memory_mib": offer.instance.resources.memory_mib,
            "spot": offer.instance.resources.spot,
            "gpu_count": len(gpus),
            "gpu_name": gpus[0].name if gpus else None,
            "gpu_memory_mib": gpus[0].memory_mib if gpus else None,
            "availability": offer.availability.value,
        })

    return results


async def list_backend_offers(session: AsyncSession, project_id: str) -> list[dict]:
    """Return available instance offers across all cloud backends for a project."""
    from dstack._internal.server.services.backends import get_project_backends

    dstack_project = await _resolve_dstack_project(session, project_id)
    backends = await get_project_backends(dstack_project)
    cloud_backends = [b for b in backends if b.TYPE.value not in ("local", "kubernetes")]
    if not cloud_backends:
        return []

    return await _fetch_offers(cloud_backends)


async def verify_backend(session: AsyncSession, project_id: str, backend_type: str) -> list[dict]:
    """Fetch offers for a single backend type on a project. Raises on failure."""
    from dstack._internal.server.services.backends import (
        get_project_backend_by_type_or_error,
    )
    from dstack._internal.core.models.backends.base import BackendType

    dstack_project = await _resolve_dstack_project(session, project_id)
    bt = BackendType(backend_type)
    backend = await get_project_backend_by_type_or_error(dstack_project, bt)
    offers = await _fetch_offers([backend])
    if not offers:
        raise RuntimeError(f"No offers returned from {backend_type} — credentials may be invalid")
    return offers


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
    backends: Optional[list[str]] = None,
    instance_types: Optional[list[str]] = None,
    regions: Optional[list[str]] = None,
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

    from dstack._internal.core.models.backends.base import BackendType

    fleet_config = FleetConfiguration(
        name=name,
        nodes=FleetNodesSpec(min=0, max=1),
        spot_policy=SpotPolicy.SPOT if use_spot else SpotPolicy.ONDEMAND,
    )
    if backends:
        fleet_config.backends = [BackendType(b) for b in backends]
    if instance_types:
        fleet_config.instance_types = instance_types
    if regions:
        fleet_config.regions = regions
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
        backends=[svc.infra] if svc.infra else None,
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
        fleet_backends = [svc.infra] if svc.infra and svc.infra != "kubernetes" else None
        fleet_instance_types = [svc.instance_type] if svc.instance_type else None
        fleet_regions = [svc.region] if svc.region else None
        await _create_fleet_for_run(
            dstack_project, name=fleet_name, gpu_spec=gpu_spec, use_spot=svc.use_spot,
            backends=fleet_backends,
            instance_types=fleet_instance_types,
            regions=fleet_regions,
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


# ── Events ─────────────────────────────────────────────────────────


async def get_run_events(
    run_name: str,
    project_name: str,
    *,
    limit: int = 100,
) -> list[dict]:
    """Fetch dstack events for a run (and its jobs)."""
    from surogate.core.compute import get_dstack_admin
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.projects import get_project_model_by_name
    from dstack._internal.server.services.runs import get_run
    from dstack._internal.server.services.events import list_events

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

            admin = get_dstack_admin()
            events = await list_events(
                session=dstack_session,
                user=admin,
                target_projects=None,
                target_users=None,
                target_fleets=None,
                target_instances=None,
                target_runs=None,
                target_jobs=None,
                target_volumes=None,
                target_gateways=None,
                target_secrets=None,
                within_projects=None,
                within_fleets=None,
                within_runs=[run.id],
                include_target_types=None,
                actors=None,
                prev_recorded_at=None,
                prev_id=None,
                limit=limit,
                ascending=True,
            )

        return [
            {
                "time": e.recorded_at.isoformat(),
                "text": e.message,
                "type": next(
                    (t.type for t in e.targets if t.type in ("run", "job")),
                    "system",
                ),
            }
            for e in events
        ]
    except Exception:
        logger.debug("Could not fetch events for run %s", run_name, exc_info=True)
        return []


# ── Instance Operations ─────────────────────────────────────────────


async def list_instances() -> list[dict]:
    """List all active dstack instances across all projects with full details."""
    import json
    from datetime import datetime, timezone
    from dstack._internal.core.models.instances import InstanceStatus
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.models import InstanceModel, JobModel, ProjectModel

    try:
        async with dstack_session_ctx() as dstack_session:
            # Flat column query — avoids relationship loading issues
            stmt = (
                sa.select(
                    InstanceModel.id,
                    InstanceModel.name,
                    InstanceModel.status,
                    InstanceModel.backend,
                    InstanceModel.region,
                    InstanceModel.price,
                    InstanceModel.offer,
                    InstanceModel.started_at,
                    InstanceModel.project_id,
                )
                .where(
                    InstanceModel.deleted == False,
                    InstanceModel.status.notin_([
                        InstanceStatus.TERMINATED,
                        InstanceStatus.TERMINATING,
                    ]),
                )
            )
            rows = (await dstack_session.execute(stmt)).all()

            # Batch-fetch project names
            project_ids = {r.project_id for r in rows}
            project_names: dict = {}
            if project_ids:
                pstmt = sa.select(ProjectModel.id, ProjectModel.name).where(
                    ProjectModel.id.in_(project_ids)
                )
                for pr in (await dstack_session.execute(pstmt)).all():
                    project_names[pr.id] = pr.name

            # Batch-fetch workload names (one job per instance)
            instance_ids = [r.id for r in rows]
            workload_map: dict = {}
            if instance_ids:
                jstmt = (
                    sa.select(JobModel.instance_id, JobModel.run_name)
                    .where(JobModel.instance_id.in_(instance_ids))
                )
                for jr in (await dstack_session.execute(jstmt)).all():
                    workload_map.setdefault(jr.instance_id, jr.run_name)

            out = []
            for r in rows:
                status = r.status.value if hasattr(r.status, "value") else str(r.status)
                backend = r.backend.value if r.backend and hasattr(r.backend, "value") else str(r.backend or "")

                # Parse offer JSON for GPU and instance type info
                gpu_label = ""
                instance_type = ""
                spot = False
                if r.offer:
                    try:
                        offer = json.loads(r.offer)
                        res = offer.get("instance", {}).get("resources", {})
                        gpus = res.get("gpus", [])
                        spot = res.get("spot", False)
                        if gpus:
                            g = gpus[0]
                            count = len(gpus)
                            gpu_label = f"{count}\u00d7 {g.get('name', '?')}"
                            mem = g.get("memory_mib")
                            if mem and mem >= 1024:
                                gpu_label += f" {mem // 1024}GB"
                        instance_type = offer.get("instance", {}).get("name", "")
                    except (json.JSONDecodeError, KeyError):
                        pass

                hours_running = 0.0
                started_at_str = None
                if r.started_at:
                    started = r.started_at.replace(tzinfo=None) if r.started_at.tzinfo else r.started_at
                    hours_running = (datetime.now(timezone.utc).replace(tzinfo=None) - started).total_seconds() / 3600
                    started_at_str = r.started_at.isoformat() + ("Z" if not r.started_at.tzinfo else "")

                price = r.price or 0.0

                out.append({
                    "id": str(r.id),
                    "name": r.name,
                    "provider": backend,
                    "region": r.region or "",
                    "instance_type": instance_type,
                    "gpu": gpu_label,
                    "status": status,
                    "workload": workload_map.get(r.id, ""),
                    "started_at": started_at_str,
                    "cost_per_hour": round(price, 2),
                    "estimated_total": round(price * hours_running, 2),
                    "spot_instance": spot,
                    "project_id": str(r.project_id),
                    "project_name": project_names.get(r.project_id, ""),
                })
            return out
    except Exception as ex:
        logger.warning("Could not list dstack instances: %s", ex, exc_info=True)
        return []


async def terminate_instance(session: AsyncSession, instance_id: str, project_name: str):
    """Terminate a dstack instance and stop the associated service/run."""
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.models import InstanceModel, JobModel
    from dstack._internal.core.models.instances import InstanceStatus
    from dstack._internal.server.services.projects import get_project_model_by_name
    from dstack._internal.server.services.runs import stop_runs

    async with dstack_session_ctx() as dstack_session:
        project = await get_project_model_by_name(
            session=dstack_session, project_name=project_name,
        )
        if project is None:
            raise ValueError(f"dstack project {project_name} not found")

        # Find the run name associated with this instance
        jstmt = sa.select(JobModel.run_name).where(JobModel.instance_id == instance_id).limit(1)
        jr = (await dstack_session.execute(jstmt)).first()
        run_name = jr.run_name if jr else None

        # Stop the dstack run (which terminates instance + job)
        if run_name:
            admin = get_dstack_admin()
            try:
                await stop_runs(
                    session=dstack_session,
                    user=admin,
                    project=project,
                    runs_names=[run_name],
                    abort=True,
                )
            except Exception as ex:
                logger.warning("stop_runs failed for instance %s: %s", instance_id, ex)
        else:
            # No run found — just mark instance for termination directly
            await dstack_session.execute(
                sa.update(InstanceModel)
                .where(InstanceModel.id == instance_id)
                .values(status=InstanceStatus.TERMINATING)
            )
            await dstack_session.commit()

    # Update the surogate serving service if one is linked to this run
    if run_name:
        from surogate.core.db.models.compute import ServingService
        result = await session.execute(
            sa.select(ServingService.id).where(ServingService.dstack_run_name == run_name)
        )
        svc_row = result.first()
        if svc_row:
            await repo.update_serving_service(
                session, svc_row.id,
                status="stopped",
            )


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


# ── Metrics ─────────────────────────────────────────────────────────


async def get_run_metrics(
    project_name: str,
    run_name: str,
    limit: int = 1,
) -> Optional[dict]:
    """Fetch the latest metrics for a dstack run.

    Returns a dict of ``{metric_name: value}`` for the most recent sample,
    or *None* if metrics are unavailable.
    """
    from dstack._internal.server.db import get_session_ctx as dstack_session_ctx
    from dstack._internal.server.services.jobs import get_run_job_model
    from dstack._internal.server.services.metrics import get_job_metrics
    from dstack._internal.server.services.projects import get_project_model_by_name

    try:
        async with dstack_session_ctx() as dstack_session:
            project = await get_project_model_by_name(
                session=dstack_session, project_name=project_name,
            )
            if project is None:
                return None

            job_model = await get_run_job_model(
                session=dstack_session,
                project=project,
                run_name=run_name,
                run_id=None,
                replica_num=0,
                job_num=0,
            )
            if job_model is None:
                return None

            job_metrics = await get_job_metrics(
                session=dstack_session,
                job_model=job_model,
                limit=limit,
            )
            if not job_metrics.metrics:
                return None

            # Flatten to {name: latest_value} for the most recent sample
            result: dict = {}
            for m in job_metrics.metrics:
                if m.values:
                    result[m.name] = m.values[0]
            return result
    except Exception:
        logger.debug(
            "Could not fetch metrics for run %s in project %s",
            run_name, project_name, exc_info=True,
        )
        return None
