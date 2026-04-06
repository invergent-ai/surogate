"""Repository functions for the compute domain."""

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.compute import (
    CloudAccount,
    CloudWorkloadType,
    ComputeNode,
    ComputePolicy,
    DeployedModel,
    ManagedJob,
    ServingService,
)


# ── ManagedJob ────────────────────────────────────────────────────────


async def create_managed_job(
    session: AsyncSession,
    *,
    name: str,
    project_id: str,
    workload_type: CloudWorkloadType,
    requested_by_id: str,
    task_yaml: str,
    dstack_run_name: Optional[str] = None,
    dstack_project_name: Optional[str] = None,
    status: str = "pending",
    accelerators: Optional[str] = None,
    cloud: Optional[str] = None,
    region: Optional[str] = None,
    use_spot: bool = False,
) -> ManagedJob:
    job = ManagedJob(
        dstack_run_name=dstack_run_name,
        dstack_project_name=dstack_project_name,
        name=name,
        project_id=project_id,
        workload_type=workload_type,
        requested_by_id=requested_by_id,
        task_yaml=task_yaml,
        status=status,
        accelerators=accelerators,
        cloud=cloud,
        region=region,
        use_spot=use_spot,
    )
    session.add(job)
    await session.commit()
    return job


async def get_managed_job(
    session: AsyncSession, job_id: str
) -> Optional[ManagedJob]:
    result = await session.execute(
        sa.select(ManagedJob).where(ManagedJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_managed_job_by_run_name(
    session: AsyncSession, run_name: str
) -> Optional[ManagedJob]:
    result = await session.execute(
        sa.select(ManagedJob).where(ManagedJob.dstack_run_name == run_name)
    )
    return result.scalar_one_or_none()


async def list_managed_jobs(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    type_filter: Optional[CloudWorkloadType] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[ManagedJob]:
    stmt = sa.select(ManagedJob).order_by(ManagedJob.created_at.desc())
    if project_id is not None:
        stmt = stmt.where(ManagedJob.project_id == project_id)
    if type_filter is not None:
        stmt = stmt.where(ManagedJob.workload_type == type_filter)
    if status is not None:
        stmt = stmt.where(ManagedJob.status == status)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_managed_job_status(
    session: AsyncSession,
    job_id: str,
    status: str,
    *,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    values: dict = {"status": status}
    if started_at is not None:
        values["started_at"] = started_at
    if completed_at is not None:
        values["completed_at"] = completed_at
    await session.execute(
        sa.update(ManagedJob).where(ManagedJob.id == job_id).values(**values)
    )
    await session.commit()


# ── ServingService ────────────────────────────────────────────────────


async def create_serving_service(
    session: AsyncSession,
    *,
    name: str,
    project_id: str,
    requested_by_id: str,
    task_yaml: str,
    status: str = "controller_init",
    endpoint: Optional[str] = None,
    accelerators: Optional[str] = None,
    infra: Optional[str] = None,
    use_spot: bool = False,
    replicas: int = 1,
    readiness_path: Optional[str] = None,
    load_balancing_policy: Optional[str] = None,
    update_mode: Optional[str] = None,
) -> ServingService:
    svc = ServingService(
        name=name,
        project_id=project_id,
        requested_by_id=requested_by_id,
        task_yaml=task_yaml,
        status=status,
        endpoint=endpoint,
        accelerators=accelerators,
        infra=infra,
        use_spot=use_spot,
        replicas=replicas,
        readiness_path=readiness_path,
        load_balancing_policy=load_balancing_policy,
        update_mode=update_mode,
    )
    session.add(svc)
    await session.commit()
    return svc


async def get_serving_service(
    session: AsyncSession, service_id: str
) -> Optional[ServingService]:
    result = await session.execute(
        sa.select(ServingService).where(ServingService.id == service_id)
    )
    return result.scalar_one_or_none()


async def get_serving_service_by_name(
    session: AsyncSession, name: str
) -> Optional[ServingService]:
    result = await session.execute(
        sa.select(ServingService).where(ServingService.name == name)
    )
    return result.scalar_one_or_none()


async def list_serving_services(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[ServingService]:
    stmt = sa.select(ServingService).order_by(ServingService.created_at.desc())
    if project_id is not None:
        stmt = stmt.where(ServingService.project_id == project_id)
    if status is not None:
        stmt = stmt.where(ServingService.status == status)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


_SERVING_TERMINAL = ("stopped", "failed", "failed_cleanup", "controller_failed")
_JOB_TERMINAL = ("completed", "failed", "cancelled")


async def list_active_serving_services(
    session: AsyncSession,
) -> list[ServingService]:
    """Return all serving services whose status is non-terminal."""
    stmt = (
        sa.select(ServingService)
        .where(ServingService.status.notin_(_SERVING_TERMINAL))
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_active_managed_jobs(
    session: AsyncSession,
) -> list[ManagedJob]:
    """Return all managed jobs whose status is non-terminal."""
    stmt = (
        sa.select(ManagedJob)
        .where(ManagedJob.status.notin_(_JOB_TERMINAL))
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def delete_serving_service(
    session: AsyncSession, service_id: str
) -> None:
    await session.execute(
        sa.delete(ServingService).where(ServingService.id == service_id)
    )
    await session.commit()


async def update_serving_service(
    session: AsyncSession,
    service_id: str,
    **values: object,
) -> None:
    await session.execute(
        sa.update(ServingService)
        .where(ServingService.id == service_id)
        .values(**values)
    )
    await session.commit()


# ── DeployedModel ────────────────────────────────────────────────────


async def create_deployed_model(
    session: AsyncSession,
    *,
    name: str,
    display_name: str,
    base_model: str,
    project_id: str,
    deployed_by_id: str,
    family: Optional[str] = None,
    param_count: Optional[str] = None,
    model_type: str = "Base",
    quantization: Optional[str] = None,
    context_window: Optional[int] = None,
    engine: Optional[str] = None,
    image: Optional[str] = None,
    hub_ref: Optional[str] = None,
    namespace: Optional[str] = None,
    serving_config: Optional[dict] = None,
    generation_defaults: Optional[dict] = None,
    serving_service_id: Optional[str] = None,
    last_deployed_at: Optional[datetime] = None,
) -> DeployedModel:
    model = DeployedModel(
        name=name,
        display_name=display_name,
        base_model=base_model,
        project_id=project_id,
        deployed_by_id=deployed_by_id,
        family=family,
        param_count=param_count,
        model_type=model_type,
        quantization=quantization,
        context_window=context_window,
        engine=engine,
        image=image,
        hub_ref=hub_ref,
        namespace=namespace,
        serving_config=serving_config,
        generation_defaults=generation_defaults,
        serving_service_id=serving_service_id,
        last_deployed_at=last_deployed_at,
    )
    session.add(model)
    await session.commit()
    return model


async def get_deployed_model(
    session: AsyncSession, model_id: str
) -> Optional[DeployedModel]:
    result = await session.execute(
        sa.select(DeployedModel).where(DeployedModel.id == model_id)
    )
    return result.scalar_one_or_none()


async def get_deployed_model_by_service(
    session: AsyncSession, service_id: str
) -> Optional[DeployedModel]:
    result = await session.execute(
        sa.select(DeployedModel).where(
            DeployedModel.serving_service_id == service_id
        )
    )
    return result.scalar_one_or_none()


async def get_deployed_model_by_name(
    session: AsyncSession, name: str
) -> Optional[DeployedModel]:
    result = await session.execute(
        sa.select(DeployedModel).where(DeployedModel.name == name)
    )
    return result.scalar_one_or_none()


async def list_deployed_models(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
) -> list[DeployedModel]:
    stmt = sa.select(DeployedModel).order_by(DeployedModel.created_at.desc())
    if project_id is not None:
        stmt = stmt.where(DeployedModel.project_id == project_id)
    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(
            sa.or_(
                DeployedModel.name.ilike(pattern),
                DeployedModel.display_name.ilike(pattern),
            )
        )
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_deployed_model(
    session: AsyncSession,
    model_id: str,
    **values: object,
) -> None:
    await session.execute(
        sa.update(DeployedModel)
        .where(DeployedModel.id == model_id)
        .values(**values)
    )
    await session.commit()


async def delete_deployed_model(
    session: AsyncSession, model_id: str
) -> None:
    await session.execute(
        sa.delete(DeployedModel).where(DeployedModel.id == model_id)
    )
    await session.commit()


# ── ComputeNode ───────────────────────────────────────────────────────


async def list_nodes(session: AsyncSession) -> list[ComputeNode]:
    result = await session.execute(sa.select(ComputeNode))
    return list(result.scalars().all())


# ── CloudAccount ──────────────────────────────────────────────────────


async def list_cloud_accounts(session: AsyncSession) -> list[CloudAccount]:
    result = await session.execute(sa.select(CloudAccount))
    return list(result.scalars().all())


# ── ComputePolicy ────────────────────────────────────────────────────


async def list_policies(session: AsyncSession) -> list[ComputePolicy]:
    result = await session.execute(sa.select(ComputePolicy))
    return list(result.scalars().all())


async def update_policy(
    session: AsyncSession, policy_id: str, *, enabled: bool
) -> Optional[ComputePolicy]:
    await session.execute(
        sa.update(ComputePolicy)
        .where(ComputePolicy.id == policy_id)
        .values(enabled=enabled)
    )
    await session.commit()
    result = await session.execute(
        sa.select(ComputePolicy).where(ComputePolicy.id == policy_id)
    )
    return result.scalar_one_or_none()
