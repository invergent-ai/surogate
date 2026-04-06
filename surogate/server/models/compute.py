"""Pydantic models for compute API routes."""

from typing import Dict, Optional

from pydantic import BaseModel


# ── Requests ──────────────────────────────────────────────────────────


class JobLaunchRequest(BaseModel):
    task_yaml: str
    name: str
    project_id: str
    workload_type: str  # training / serving / eval
    accelerators: Optional[str] = None
    cloud: Optional[str] = None
    use_spot: bool = False


class ServingServiceLaunchRequest(BaseModel):
    task_yaml: str
    service_name: str
    project_id: str
    accelerators: Optional[str] = None
    infra: Optional[str] = None
    use_spot: bool = False
    replicas: int = 1
    readiness_path: Optional[str] = None
    load_balancing_policy: Optional[str] = None


class PolicyToggleRequest(BaseModel):
    enabled: bool


# ── Responses ─────────────────────────────────────────────────────────


class JobResponse(BaseModel):
    id: str
    name: str
    type: str  # training / serving / eval
    method: str  # SFT / GRPO / DPO / —
    status: str  # running / queued / provisioning / completed / failed / cancelled
    gpu: Optional[str] = None
    gpu_count: int = 0
    location: str = "local"
    node: Optional[str] = None
    eta: Optional[str] = None
    started_at: Optional[str] = None
    requested_by: Optional[str] = None
    project: Optional[str] = None
    cloud: Optional[str] = None
    region: Optional[str] = None
    use_spot: bool = False
    dstack_run_name: Optional[str] = None

    class Config:
        orm_mode = True


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    status_counts: dict[str, int]


class ServingServiceResponse(BaseModel):
    id: str
    name: str
    status: str
    endpoint: Optional[str] = None
    accelerators: Optional[str] = None
    infra: Optional[str] = None
    use_spot: bool = False
    num_replicas: int = 1
    readiness_path: Optional[str] = None
    dstack_run_name: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    requested_by: Optional[str] = None
    project: Optional[str] = None

    class Config:
        orm_mode = True


class ServingServiceListResponse(BaseModel):
    services: list[ServingServiceResponse]
    total: int
    status_counts: dict[str, int]


class GpuInfo(BaseModel):
    type: str
    count: int
    used: int
    utilization: float


class CpuInfo(BaseModel):
    cores: int
    used: int
    utilization: float


class MemInfo(BaseModel):
    total: int
    used: int
    unit: str = "Gi"


class WorkloadSlot(BaseModel):
    name: str
    type: str
    gpu: int


class NodeResponse(BaseModel):
    id: str
    hostname: str
    pool: str
    status: str
    gpu: Optional[GpuInfo] = None
    cpu: CpuInfo
    mem: MemInfo
    workloads: list[WorkloadSlot]

    class Config:
        orm_mode = True
        use_enum_values = True


class CloudAccountResponse(BaseModel):
    provider: str
    name: str
    status: str
    quota_gpu: int = 0
    used_gpu: int = 0
    regions: list[str] = []
    monthly_budget: float = 0
    monthly_spend: float = 0

    class Config:
        orm_mode = True
        use_enum_values = True


class CloudInstanceResponse(BaseModel):
    id: str
    provider: str
    region: str
    instance_type: str
    gpu: str
    status: str
    workload: str
    started_at: Optional[str] = None
    cost_per_hour: float = 0
    estimated_total: float = 0
    spot_instance: bool = False
    spot_savings: Optional[str] = None
    auto_terminate: Optional[str] = None

    class Config:
        orm_mode = True
        use_enum_values = True


class PolicyResponse(BaseModel):
    id: str
    name: str
    enabled: bool
    trigger: str
    action: str
    cooldown: str
    last_triggered: Optional[str] = None
    trigger_count: int = 0

    class Config:
        orm_mode = True


class CostByTypeItem(BaseModel):
    type: str
    cost: float
    pct: float
    color: str


class CostByProjectItem(BaseModel):
    project: str
    cost: float
    pct: float
    color: str


class DailySpendItem(BaseModel):
    day: int
    value: float


class CostResponse(BaseModel):
    daily_spend: list[DailySpendItem]
    by_type: list[CostByTypeItem]
    by_project: list[CostByProjectItem]
    monthly_total: float


class OverviewResponse(BaseModel):
    local_gpu_used: int
    local_gpu_total: int
    local_node_count: int
    cloud_gpu_total: int
    cloud_instance_count: int
    cloud_hourly_cost: float
    monthly_spend: float
    monthly_budget: float
    queue_running: int
    queue_queued: int

class K8NodeMetricsResponse(BaseModel):
    timestamp: int
    node_name: str
    total_memory_bytes: Optional[int] = None
    free_memory_bytes: Optional[int] = None
    cpu_utilization_percent: Optional[float] = None
    
class K8NodeResponse(BaseModel):
    name: str
    accelerator_type: Optional[str]
    accelerator_count: int = 0
    accelerator_available: int = 0
    # CPU count (total CPUs available on the node)
    cpu_count: Optional[float] = None
    # Memory in GB (total memory available on the node)
    memory_gb: Optional[float] = None
    # Whether the node is ready (all conditions are satisfied)
    is_ready: bool = True
    
    metrics: Optional[K8NodeMetricsResponse] = None
    
    