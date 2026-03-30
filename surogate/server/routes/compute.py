"""Compute API routes — managed jobs, nodes, cloud, costs, policies."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute import _initialized as skypilot_initialized
from surogate.core.compute import service as compute_service
from surogate.core.db.engine import get_session
from surogate.core.db.repository import compute as compute_repo
from surogate.server.auth.authentication import get_current_subject
from surogate.server.models.compute import (
    CloudAccountResponse,
    HealthResponse,
    JobLaunchRequest,
    JobListResponse,
    JobResponse,
    NodeResponse,
    OverviewResponse,
    PolicyResponse,
    PolicyToggleRequest,
)

router = APIRouter()


# ── Health ───────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Check whether SkyPilot has been initialized."""
    version = None
    try:
        import sky
        version = getattr(sky, "__version__", None)
    except Exception:
        pass
    return HealthResponse(initialized=skypilot_initialized, version=version)


# ── Overview ─────────────────────────────────────────────────────────


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Aggregated dashboard KPIs for the Overview tab."""
    nodes = await compute_repo.list_nodes(session)
    accounts = await compute_repo.list_cloud_accounts(session)
    jobs_data = await compute_service.list_jobs(session, limit=200)

    local_gpu_total = sum(n.gpu_total for n in nodes)
    local_gpu_used = sum(n.gpu_used for n in nodes)

    cloud_instances = []
    try:
        cloud_instances = await compute_service.get_cluster_status()
    except Exception:
        pass

    cloud_gpu_total = 0
    cloud_hourly_cost = 0.0
    for ci in cloud_instances:
        gpu_count = getattr(ci, "gpu_count", 0) or 0
        cloud_gpu_total += gpu_count

    monthly_spend = sum(a.monthly_spend for a in accounts)
    monthly_budget = sum(a.monthly_budget for a in accounts)

    sc = jobs_data.get("status_counts", {})

    return OverviewResponse(
        local_gpu_used=local_gpu_used,
        local_gpu_total=local_gpu_total,
        local_node_count=len(nodes),
        cloud_gpu_total=cloud_gpu_total,
        cloud_instance_count=len(cloud_instances),
        cloud_hourly_cost=cloud_hourly_cost,
        monthly_spend=monthly_spend,
        monthly_budget=monthly_budget,
        queue_running=sc.get("running", 0),
        queue_queued=sc.get("queued", 0),
    )


# ── Managed Jobs ─────────────────────────────────────────────────────


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """List managed jobs with optional filters."""
    data = await compute_service.list_jobs(
        session,
        project_id=project_id,
        type_filter=type,
        status_filter=status,
        limit=limit,
    )
    jobs = [
        JobResponse(
            id=j.id,
            name=j.name,
            type=j.workload_type.value,
            method="—",
            status=j.status,
            gpu=j.accelerators,
            gpu_count=0,
            location=j.cloud or "local",
            node=None,
            eta=None,
            started_at=j.started_at.isoformat() if j.started_at else None,
            requested_by=j.requested_by_id,
            project=j.project_id,
            cloud=j.cloud,
            region=j.region,
            use_spot=j.use_spot,
            skypilot_job_id=j.skypilot_job_id,
        )
        for j in data["jobs"]
    ]
    return JobListResponse(
        jobs=jobs,
        total=data["total"],
        status_counts=data["status_counts"],
    )


@router.post("/jobs", response_model=JobResponse)
async def launch_job(
    req: JobLaunchRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Submit a new managed job."""
    job = await compute_service.launch_job(
        session,
        task_yaml=req.task_yaml,
        name=req.name,
        project_id=req.project_id,
        workload_type=req.workload_type,
        requested_by_id=current_subject,
        accelerators=req.accelerators,
        cloud=req.cloud,
        use_spot=req.use_spot,
    )
    return JobResponse(
        id=job.id,
        name=job.name,
        type=job.workload_type.value,
        method="—",
        status=job.status,
        gpu=job.accelerators,
        location=job.cloud or "local",
        cloud=job.cloud,
        region=job.region,
        use_spot=job.use_spot,
        skypilot_job_id=job.skypilot_job_id,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Cancel a managed job."""
    try:
        await compute_service.cancel_job(session, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "cancelled"}


# ── Cluster Nodes ────────────────────────────────────────────────────


@router.get("/nodes", response_model=list[NodeResponse])
async def list_nodes(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """List local cluster nodes."""
    nodes = await compute_repo.list_nodes(session)
    return [
        NodeResponse(
            id=n.id,
            hostname=n.hostname,
            pool=n.pool.value,
            status=n.status.value,
            gpu=None,
            cpu={"cores": n.cpu_cores, "used": 0, "utilization": n.cpu_used_percent},
            mem={"total": 0, "used": 0, "unit": "Gi"},
            workloads=[],
        )
        for n in nodes
    ]


# ── Cloud ────────────────────────────────────────────────────────────


@router.get("/cloud/accounts", response_model=list[CloudAccountResponse])
async def list_cloud_accounts(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """List cloud provider accounts."""
    accounts = await compute_repo.list_cloud_accounts(session)
    return [
        CloudAccountResponse(
            provider=a.provider.value,
            name=f"{a.provider.value.upper()} Account",
            status=a.status.value,
            quota_gpu=a.gpu_quota_total,
            used_gpu=a.gpu_quota_used,
            regions=a.regions or [],
            monthly_budget=a.monthly_budget,
            monthly_spend=a.monthly_spend,
        )
        for a in accounts
    ]


@router.get("/cloud/instances")
async def list_cloud_instances(
    current_subject: str = Depends(get_current_subject),
):
    """List active SkyPilot clusters / cloud instances."""
    try:
        clusters = await compute_service.get_cluster_status()
    except Exception:
        clusters = []
    return clusters


@router.post("/cloud/instances/{cluster_name}/terminate")
async def terminate_cloud_instance(
    cluster_name: str,
    current_subject: str = Depends(get_current_subject),
):
    """Tear down a SkyPilot cluster."""
    try:
        await compute_service.terminate_cluster(cluster_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "terminated"}


# ── Costs ────────────────────────────────────────────────────────────


@router.get("/costs")
async def get_costs(
    current_subject: str = Depends(get_current_subject),
    days: int = Query(30, le=365),
):
    """Cost report from SkyPilot."""
    try:
        report = await compute_service.get_cost_report(days=days)
    except Exception:
        report = []
    return report


# ── Policies ─────────────────────────────────────────────────────────


@router.get("/policies", response_model=list[PolicyResponse])
async def list_policies(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """List compute auto-scaling policies."""
    policies = await compute_repo.list_policies(session)
    return [
        PolicyResponse(
            id=p.id,
            name=p.name,
            enabled=p.enabled,
            trigger=p.condition,
            action=p.action,
            cooldown=p.cooldown,
            last_triggered=p.last_triggered_at.isoformat() if p.last_triggered_at else None,
            trigger_count=p.trigger_count,
        )
        for p in policies
    ]


@router.patch("/policies/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str,
    req: PolicyToggleRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Toggle or update a compute policy."""
    policy = await compute_repo.update_policy(session, policy_id, enabled=req.enabled)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return PolicyResponse(
        id=policy.id,
        name=policy.name,
        enabled=policy.enabled,
        trigger=policy.condition,
        action=policy.action,
        cooldown=policy.cooldown,
        last_triggered=policy.last_triggered_at.isoformat() if policy.last_triggered_at else None,
        trigger_count=policy.trigger_count,
    )
