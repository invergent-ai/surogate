"""Compute domain: ComputeNode, CloudInstance, CloudAccount, ComputePolicy."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from surogate.core.db.base import Base, UUIDMixin


# ── Enums ──────────────────────────────────────────────────────────────


class ComputePool(enum.Enum):
    training = "training"
    serving = "serving"
    system = "system"


class ComputeNodeStatus(enum.Enum):
    active = "active"
    cordoned = "cordoned"
    error = "error"


class CloudProvider(enum.Enum):
    aws = "aws"
    gcp = "gcp"
    azure = "azure"


class CloudInstanceStatus(enum.Enum):
    running = "running"
    provisioning = "provisioning"
    terminated = "terminated"


class CloudAccountStatus(enum.Enum):
    connected = "connected"
    disconnected = "disconnected"


class CloudWorkloadType(enum.Enum):
    training = "training"
    serving = "serving"
    eval = "eval"


class LocalTaskType(enum.Enum):
    import_model = "import_model"
    import_dataset = "import_dataset"
    local_inference = "local_inference"
    local_training = "local_training"


class LocalTaskStatus(enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# ── Models ─────────────────────────────────────────────────────────────


class ComputeNode(Base):
    __tablename__ = "compute_nodes"

    id: Mapped[str] = mapped_column(sa.String(255), primary_key=True)
    hostname: Mapped[str] = mapped_column(sa.String(255))
    pool: Mapped[ComputePool] = mapped_column(sa.Enum(ComputePool))
    status: Mapped[ComputeNodeStatus] = mapped_column(sa.Enum(ComputeNodeStatus))
    gpu_type: Mapped[str] = mapped_column(sa.String(128))
    gpu_total: Mapped[int] = mapped_column(sa.Integer)
    gpu_used: Mapped[int] = mapped_column(sa.Integer, default=0)
    cpu_cores: Mapped[int] = mapped_column(sa.Integer)
    cpu_used_percent: Mapped[float] = mapped_column(sa.Float, default=0.0)
    memory_total: Mapped[str] = mapped_column(sa.String(32))
    memory_used: Mapped[str] = mapped_column(sa.String(32), default="0")


class CloudInstance(UUIDMixin, Base):
    __tablename__ = "cloud_instances"

    provider: Mapped[CloudProvider] = mapped_column(sa.Enum(CloudProvider))
    region: Mapped[str] = mapped_column(sa.String(64))
    instance_type: Mapped[str] = mapped_column(sa.String(64))
    gpu_type: Mapped[str] = mapped_column(sa.String(128))
    gpu_count: Mapped[int] = mapped_column(sa.Integer)
    status: Mapped[CloudInstanceStatus] = mapped_column(
        sa.Enum(CloudInstanceStatus)
    )
    spot: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    cost_per_hour: Mapped[float] = mapped_column(sa.Float)
    workload_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), nullable=True
    )
    workload_type: Mapped[Optional[CloudWorkloadType]] = mapped_column(
        sa.Enum(CloudWorkloadType), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    auto_terminate_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )


class CloudAccount(UUIDMixin, Base):
    __tablename__ = "cloud_accounts"

    provider: Mapped[CloudProvider] = mapped_column(sa.Enum(CloudProvider))
    status: Mapped[CloudAccountStatus] = mapped_column(
        sa.Enum(CloudAccountStatus)
    )
    gpu_quota_used: Mapped[int] = mapped_column(sa.Integer, default=0)
    gpu_quota_total: Mapped[int] = mapped_column(sa.Integer, default=0)
    monthly_spend: Mapped[float] = mapped_column(sa.Float, default=0.0)
    monthly_budget: Mapped[float] = mapped_column(sa.Float, default=0.0)
    regions: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)


class ManagedJob(UUIDMixin, Base):
    """Platform metadata for a SkyPilot managed job.

    SkyPilot's own SQLite DB tracks job/cluster state.  This model stores
    Surogate-specific context: which project owns the job, who requested it,
    what type of workload it is, and the original task YAML for audit/replay.
    """

    __tablename__ = "managed_jobs"

    skypilot_job_id: Mapped[Optional[int]] = mapped_column(
        sa.Integer, index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    workload_type: Mapped[CloudWorkloadType] = mapped_column(
        sa.Enum(CloudWorkloadType)
    )
    requested_by_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), index=True
    )
    task_yaml: Mapped[str] = mapped_column(sa.Text)
    status: Mapped[str] = mapped_column(sa.String(32), default="pending")
    accelerators: Mapped[Optional[str]] = mapped_column(
        sa.String(128), nullable=True
    )
    cloud: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True
    )
    region: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True
    )
    use_spot: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )


class LocalTask(UUIDMixin, Base):
    """A local long-running task executed as a subprocess.

    Used for operations like importing models/datasets from HuggingFace,
    running local inference, or local training.  Each task runs in its own
    process with stdout captured to a log file under ~/.surogate/tasks/.
    """

    __tablename__ = "local_tasks"

    name: Mapped[str] = mapped_column(sa.String(255))
    task_type: Mapped[LocalTaskType] = mapped_column(sa.Enum(LocalTaskType))
    status: Mapped[LocalTaskStatus] = mapped_column(
        sa.Enum(LocalTaskStatus), default=LocalTaskStatus.pending
    )
    pid: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    exit_code: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    progress: Mapped[Optional[str]] = mapped_column(
        sa.String(255), nullable=True
    )
    log_path: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    requested_by_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), index=True
    )
    params: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )


class ServingService(UUIDMixin, Base):
    """Platform metadata for a SkyPilot serving service.

    SkyPilot Serve manages the controller, load balancer, and replicas.
    This model stores Surogate-specific context: project ownership,
    who launched it, original task YAML, and replica/autoscaling config.
    """

    __tablename__ = "serving_services"

    name: Mapped[str] = mapped_column(sa.String(255), unique=True)
    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    requested_by_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), index=True
    )
    task_yaml: Mapped[str] = mapped_column(sa.Text)
    status: Mapped[str] = mapped_column(sa.String(32), default="controller_init")
    endpoint: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    accelerators: Mapped[Optional[str]] = mapped_column(
        sa.String(128), nullable=True
    )
    cloud: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True
    )
    region: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True
    )
    use_spot: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    min_replicas: Mapped[int] = mapped_column(sa.Integer, default=1)
    max_replicas: Mapped[Optional[int]] = mapped_column(
        sa.Integer, nullable=True
    )
    readiness_path: Mapped[Optional[str]] = mapped_column(
        sa.String(255), nullable=True
    )
    load_balancing_policy: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True
    )
    update_mode: Mapped[Optional[str]] = mapped_column(
        sa.String(32), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    terminated_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )


class DeployedModelStatus(enum.Enum):
    deploying = "deploying"
    serving = "serving"
    error = "error"
    stopped = "stopped"


class DeployedModel(UUIDMixin, Base):
    """A model deployment linked to a ServingService.

    Captures model-specific metadata (identity, config, generation defaults)
    while the linked ServingService handles infrastructure (SkyPilot, replicas,
    cloud placement).  Status is derived from the ServingService state.
    """

    __tablename__ = "deployed_models"

    name: Mapped[str] = mapped_column(sa.String(255), unique=True)
    display_name: Mapped[str] = mapped_column(sa.String(512))
    base_model: Mapped[str] = mapped_column(sa.String(512))
    family: Mapped[Optional[str]] = mapped_column(
        sa.String(128), nullable=True
    )
    param_count: Mapped[Optional[str]] = mapped_column(
        sa.String(32), nullable=True
    )
    model_type: Mapped[str] = mapped_column(
        sa.String(64), default="Base"
    )
    quantization: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True
    )
    context_window: Mapped[Optional[int]] = mapped_column(
        sa.Integer, nullable=True
    )
    engine: Mapped[Optional[str]] = mapped_column(
        sa.String(128), nullable=True
    )
    image: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    namespace: Mapped[Optional[str]] = mapped_column(
        sa.String(128), nullable=True
    )
    serving_config: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    generation_defaults: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    serving_service_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("serving_services.id"), nullable=True, index=True
    )
    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    deployed_by_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    last_deployed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )


class ComputePolicy(UUIDMixin, Base):
    __tablename__ = "compute_policies"

    name: Mapped[str] = mapped_column(sa.String(255))
    enabled: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    condition: Mapped[str] = mapped_column(sa.Text)
    action: Mapped[str] = mapped_column(sa.Text)
    cooldown: Mapped[str] = mapped_column(sa.String(64))
    trigger_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )


__all__ = [
    "ComputePool",
    "ComputeNodeStatus",
    "CloudProvider",
    "CloudInstanceStatus",
    "CloudAccountStatus",
    "CloudWorkloadType",
    "ComputeNode",
    "CloudInstance",
    "CloudAccount",
    "ManagedJob",
    "LocalTaskType",
    "LocalTaskStatus",
    "LocalTask",
    "ServingService",
    "DeployedModelStatus",
    "DeployedModel",
    "ComputePolicy",
]
