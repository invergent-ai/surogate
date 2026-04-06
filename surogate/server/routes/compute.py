"""Compute API routes — managed jobs, nodes, cloud, policies."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from prometheus_api_client.prometheus_connect import PrometheusConnect
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute import dstack as dstack_service
from surogate.core.db.engine import get_session
from surogate.core.db.repository import compute as compute_repo
from surogate.server.auth.authentication import get_current_subject, get_current_user_id
from surogate.server.models.compute import (
    CloudAccountResponse,
    K8NodeResponse,
    K8NodeMetricsResponse,
    JobLaunchRequest,
    JobListResponse,
    JobResponse,
    NodeResponse,
    OverviewResponse,
    PolicyResponse,
    PolicyToggleRequest,
)

router = APIRouter()


# ── Kubernetes ───────────────────────────────────────────────────────────
@router.get("/nodes", response_model=list[K8NodeResponse])
async def get_local_node_metrics(
    request: Request,
):
    import surogate.core.compute.kubernetes as k8s

    if k8s.k8_nodes is None:
        return []

    prom = PrometheusConnect(url=request.app.state.config.prometheus_endpoint, disable_ssl=True)
    responses: list[K8NodeResponse] = []

    for node in k8s.k8_nodes.node_info_dict.values():
        node_available_mem = prom.custom_query(query=f"node_memory_MemAvailable_bytes{{node='{node.name}'}}")
        node_total_mem = prom.custom_query(query=f"node_memory_MemTotal_bytes{{node='{node.name}'}}")
        node_cpu_util = prom.custom_query(query=f'1 - avg by(node) (irate(node_cpu_seconds_total{{mode="idle", node="{node.name}"}}[1m]))')

        metrics = K8NodeMetricsResponse(node_name=node.name, timestamp=int(datetime.now().timestamp()))
        if len(node_available_mem) > 0:
            value = node_available_mem[0].get('value')
            if value and len(value) > 1:
                metrics.free_memory_bytes = int(value[1])

        if len(node_total_mem) > 0:
            value = node_total_mem[0].get('value')
            if value and len(value) > 1:
                metrics.total_memory_bytes = int(value[1])

        if len(node_cpu_util) > 0:
            value = node_cpu_util[0].get('value')
            if value and len(value) > 1:
                metrics.cpu_utilization_percent = float(value[1]) * 100

        responses.append(K8NodeResponse(
            name=node.name,
            accelerator_type=node.accelerator_type,
            accelerator_count=node.accelerator_count,
            accelerator_available=node.accelerator_available,
            cpu_count=node.cpu_count,
            memory_gb=node.memory_gb,
            is_ready=node.is_ready,
            metrics=metrics
        ))

    return responses


# ── Overview ─────────────────────────────────────────────────────────


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Aggregated dashboard KPIs for the Overview tab."""
    nodes = await compute_repo.list_nodes(session)
    accounts = await compute_repo.list_cloud_accounts(session)
    jobs_data = await dstack_service.list_tasks(session, limit=200)

    local_gpu_total = sum(n.gpu_total for n in nodes)
    local_gpu_used = sum(n.gpu_used for n in nodes)

    # Cloud instances from dstack
    cloud_instances = await dstack_service.list_instances()
    cloud_gpu_total = 0
    cloud_hourly_cost = 0.0

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
    data = await dstack_service.list_tasks(
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
            dstack_run_name=j.dstack_run_name,
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
    current_user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    """Submit a new managed job."""
    job = await dstack_service.submit_task(
        session,
        name=req.name,
        project_id=req.project_id,
        workload_type=req.workload_type,
        requested_by_id=current_user_id,
        image="python:3.11",  # TODO: make configurable via request
        commands=[req.task_yaml],  # task_yaml used as the command for now
        accelerators=req.accelerators,
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
        dstack_run_name=job.dstack_run_name,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Cancel a managed job."""
    try:
        await dstack_service.cancel_task(session, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "cancelled"}


# ── Instances ───────────────────────────────────────────────────────


@router.get("/cloud/instances")
async def list_cloud_instances(
    current_subject: str = Depends(get_current_subject),
):
    """List active dstack instances."""
    return await dstack_service.list_instances()


@router.post("/cloud/instances/{instance_id}/terminate")
async def terminate_cloud_instance(
    instance_id: str,
    project_name: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    """Terminate a dstack instance."""
    if not project_name:
        raise HTTPException(status_code=400, detail="project_name query param required")
    try:
        await dstack_service.terminate_instance(instance_id, project_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "terminated"}


# ── Cloud Accounts ──────────────────────────────────────────────────

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
