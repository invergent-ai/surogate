"""Compute API routes — managed jobs, nodes, cloud, costs, policies."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from prometheus_api_client.prometheus_connect import PrometheusConnect
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute import sky as sky_service
from surogate.core.db.engine import get_session
from surogate.core.db.repository import compute as compute_repo
from surogate.server.auth.authentication import get_current_subject
import surogate.core.compute.kubernetes as k8s
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
    ServingServiceLaunchRequest,
    ServingServiceListResponse,
    ServingServiceReplicaResponse,
    ServingServiceResponse,
    ServingServiceUpdateRequest,
)

router = APIRouter()


# ── Kubernetes ───────────────────────────────────────────────────────────
@router.get("/nodes", response_model=list[K8NodeResponse])
async def get_local_node_metrics(
    request: Request,
):
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
            total=node.total,
            free=node.free,
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
    jobs_data = await sky_service.list_jobs(session, limit=200)

    local_gpu_total = sum(n.gpu_total for n in nodes)
    local_gpu_used = sum(n.gpu_used for n in nodes)

    cloud_instances = []
    try:
        cloud_instances = await sky_service.get_cluster_status()
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


# ── Serving Services ────────────────────────────────────────────────


@router.get("/services", response_model=ServingServiceListResponse)
async def list_serving_services(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """List serving services with optional filters."""
    data = await sky_service.list_serving_services(
        session,
        project_id=project_id,
        status_filter=status,
        limit=limit,
    )
    services = [
        ServingServiceResponse(
            id=s.id,
            name=s.name,
            status=s.status,
            endpoint=s.endpoint,
            accelerators=s.accelerators,
            cloud=s.cloud,
            region=s.region,
            use_spot=s.use_spot,
            min_replicas=s.min_replicas,
            max_replicas=s.max_replicas,
            readiness_path=s.readiness_path,
            load_balancing_policy=s.load_balancing_policy,
            update_mode=s.update_mode,
            created_at=s.created_at.isoformat() if s.created_at else None,
            started_at=s.started_at.isoformat() if s.started_at else None,
            requested_by=s.requested_by_id,
            project=s.project_id,
        )
        for s in data["services"]
    ]
    return ServingServiceListResponse(
        services=services,
        total=data["total"],
        status_counts=data["status_counts"],
    )


@router.post("/services", response_model=ServingServiceResponse)
async def launch_serving_service(
    req: ServingServiceLaunchRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Launch a new serving service."""
    svc = await sky_service.launch_serving_service(
        session,
        task_yaml=req.task_yaml,
        service_name=req.service_name,
        project_id=req.project_id,
        requested_by_id=current_subject,
        accelerators=req.accelerators,
        cloud=req.cloud,
        use_spot=req.use_spot,
        min_replicas=req.min_replicas,
        max_replicas=req.max_replicas,
        readiness_path=req.readiness_path,
        load_balancing_policy=req.load_balancing_policy,
    )
    return ServingServiceResponse(
        id=svc.id,
        name=svc.name,
        status=svc.status,
        endpoint=svc.endpoint,
        accelerators=svc.accelerators,
        cloud=svc.cloud,
        region=svc.region,
        use_spot=svc.use_spot,
        min_replicas=svc.min_replicas,
        max_replicas=svc.max_replicas,
        readiness_path=svc.readiness_path,
        load_balancing_policy=svc.load_balancing_policy,
    )


@router.get("/services/{service_id}/status", response_model=ServingServiceResponse)
async def get_serving_service_status(
    service_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Get detailed status for a serving service including replicas."""
    svc = await compute_repo.get_serving_service(session, service_id)
    if svc is None:
        raise HTTPException(status_code=404, detail="Service not found")

    sky_info = await sky_service.get_serving_service_status(svc.name)
    replicas = []
    if sky_info and "replica_info" in sky_info:
        replicas = [
            ServingServiceReplicaResponse(
                replica_id=r["replica_id"],
                name=r.get("name"),
                status=str(r.get("status", "unknown")),
                version=r.get("version"),
                launched_at=str(r["launched_at"]) if r.get("launched_at") else None,
                endpoint=r.get("endpoint"),
            )
            for r in sky_info["replica_info"]
        ]

    return ServingServiceResponse(
        id=svc.id,
        name=svc.name,
        status=svc.status,
        endpoint=svc.endpoint,
        accelerators=svc.accelerators,
        cloud=svc.cloud,
        region=svc.region,
        use_spot=svc.use_spot,
        min_replicas=svc.min_replicas,
        max_replicas=svc.max_replicas,
        readiness_path=svc.readiness_path,
        load_balancing_policy=svc.load_balancing_policy,
        update_mode=svc.update_mode,
        replicas=replicas,
        created_at=svc.created_at.isoformat() if svc.created_at else None,
        started_at=svc.started_at.isoformat() if svc.started_at else None,
        requested_by=svc.requested_by_id,
        project=svc.project_id,
    )


@router.put("/services/{service_id}", response_model=ServingServiceResponse)
async def update_serving_service(
    service_id: str,
    req: ServingServiceUpdateRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Update a serving service with a new task configuration."""
    try:
        svc = await sky_service.update_serving_service(
            session, service_id, task_yaml=req.task_yaml, mode=req.mode
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ServingServiceResponse(
        id=svc.id,
        name=svc.name,
        status=svc.status,
        endpoint=svc.endpoint,
        accelerators=svc.accelerators,
        cloud=svc.cloud,
        region=svc.region,
        use_spot=svc.use_spot,
        min_replicas=svc.min_replicas,
        max_replicas=svc.max_replicas,
        update_mode=req.mode,
    )


@router.delete("/services/{service_id}")
async def terminate_serving_service(
    service_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    purge: bool = Query(False),
):
    """Tear down a serving service."""
    try:
        await sky_service.terminate_serving_service(session, service_id, purge=purge)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "shutting_down"}


@router.post("/services/{service_name}/replicas/{replica_id}/terminate")
async def terminate_replica(
    service_name: str,
    replica_id: int,
    current_subject: str = Depends(get_current_subject),
    purge: bool = Query(False),
):
    """Terminate a specific replica of a serving service."""
    try:
        await sky_service.terminate_serving_service_replica(
            service_name, replica_id, purge=purge
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "terminated"}


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
    data = await sky_service.list_jobs(
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
    job = await sky_service.launch_job(
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
        await sky_service.cancel_job(session, job_id)
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
        clusters = await sky_service.get_cluster_status()
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
        await sky_service.terminate_cluster(cluster_name)
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
        report = await sky_service.get_cost_report(days=days)
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
