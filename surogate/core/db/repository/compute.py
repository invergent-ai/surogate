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
    ManagedJob,
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
    skypilot_job_id: Optional[int] = None,
    status: str = "pending",
    accelerators: Optional[str] = None,
    cloud: Optional[str] = None,
    region: Optional[str] = None,
    use_spot: bool = False,
) -> ManagedJob:
    job = ManagedJob(
        skypilot_job_id=skypilot_job_id,
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


async def get_managed_job_by_skypilot_id(
    session: AsyncSession, sky_job_id: int
) -> Optional[ManagedJob]:
    result = await session.execute(
        sa.select(ManagedJob).where(ManagedJob.skypilot_job_id == sky_job_id)
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
