"""Pydantic models for deployed-model API routes."""

from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Requests ──────────────────────────────────────────────────────────


class DeployedModelCreateRequest(BaseModel):
    name: str
    display_name: str
    base_model: str
    project_id: str
    family: Optional[str] = None
    param_count: Optional[str] = None
    model_type: str = "Base"
    quantization: Optional[str] = None
    context_window: Optional[int] = None
    engine: Optional[str] = None
    image: Optional[str] = None
    hub_ref: Optional[str] = None
    namespace: Optional[str] = None
    source: Optional[str] = None
    serving_config: Optional[dict] = None
    generation_defaults: Optional[dict] = None


class DeployedModelUpdateRequest(BaseModel):
    engine: Optional[str] = None
    accelerators: Optional[str] = None
    infra: Optional[str] = None
    instance_type: Optional[str] = None
    region: Optional[str] = None
    use_spot: Optional[bool] = None
    serving_config: Optional[dict] = None
    generation_defaults: Optional[dict] = None


class DeployedModelScaleRequest(BaseModel):
    replicas: int


# ── Response sub-models ───────────────────────────────────────────────


class ReplicaInfo(BaseModel):
    current: int
    desired: int


class GpuInfo(BaseModel):
    type: str
    count: int
    utilization: float


class VramInfo(BaseModel):
    used: str
    total: str
    pct: float


class ConnectedAgentInfo(BaseModel):
    name: str
    status: str
    rps: float


class FineTuneInfo(BaseModel):
    name: str
    method: str
    dataset: str
    date: str
    status: str
    loss: str
    hub_ref: str


class MetricsHistoryInfo(BaseModel):
    tps: list[float] = []
    latency: list[float] = []
    gpu: list[float] = []
    queue: list[float] = []


class EventInfo(BaseModel):
    time: str
    text: str
    type: str


# ── Main response ────────────────────────────────────────────────────


class DeployedModelResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: str
    base: str
    project_id: str
    family: str
    param_count: str
    type: str
    quantization: str
    context_window: int
    status: str
    engine: str
    replicas: ReplicaInfo
    gpu: GpuInfo
    vram: VramInfo
    cpu: str
    mem: str
    mem_limit: str
    tps: float
    p50: str
    p95: str
    p99: str
    queue_depth: int
    batch_size: str
    tokens_in_24h: str
    tokens_out_24h: str
    requests_24h: int
    error_rate: str
    uptime: str
    last_deployed: str
    deployed_by: str
    namespace: str
    project_color: str
    endpoint: str
    image: str
    hub_ref: str
    infra: Optional[str] = None
    source: Optional[str] = None
    use_spot: bool = False
    connected_agents: list[ConnectedAgentInfo] = []
    serving_config: Optional[dict] = None
    generation_defaults: Optional[dict] = None
    fine_tunes: list[FineTuneInfo] = []
    metrics_history: MetricsHistoryInfo = MetricsHistoryInfo()
    events: list[EventInfo] = []

    model_config = ConfigDict(from_attributes=True)


class DeployedModelListResponse(BaseModel):
    models: list[DeployedModelResponse]
    total: int
    status_counts: dict[str, int]


class ModelLogsResponse(BaseModel):
    model_id: str
    lines: list[str]


class ModelEventsResponse(BaseModel):
    model_id: str
    events: list[EventInfo]
