"""Service layer for deployed models.

Bridges DeployedModel (model identity + config) with ServingService
(infrastructure / SkyPilot).  Status is derived from the linked
ServingService state rather than stored on the model itself.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute import sky as sky_service
from surogate.core.config.server_config import ServerConfig
from surogate.core.db.models.compute import DeployedModel, ServingService
from surogate.core.db.repository import compute as repo
from surogate.core.hub import lakefs
from surogate.core.hub.model_info import resolve_from_huggingface, resolve_from_lakefs
from surogate.server.models.models import (
    DeployedModelResponse,
    GpuInfo,
    MetricsHistoryInfo,
    ReplicaInfo,
    VramInfo,
)
from surogate.utils.logger import get_logger

logger = get_logger()


# ── Status derivation ────────────────────────────────────────────────

_SERVING_TO_MODEL_STATUS: dict[str, str] = {
    "controller_init": "deploying",
    "replica_init": "deploying",
    "ready": "serving",
    "controller_failed": "error",
    "failed": "error",
    "failed_cleanup": "error",
    "no_replica": "error",
    "shutting_down": "stopped",
}


def _derive_status(svc: Optional[ServingService]) -> str:
    if svc is None:
        return "stopped"
    return _SERVING_TO_MODEL_STATUS.get(svc.status, "stopped")


def _relative_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "\u2014"
    now = datetime.now(timezone.utc)
    naive = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    delta = now - naive
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _uptime(svc: Optional[ServingService]) -> str:
    if svc is None or svc.started_at is None:
        return "\u2014"
    if svc.status not in ("ready",):
        return "\u2014"
    now = datetime.now(timezone.utc)
    started = svc.started_at.replace(tzinfo=timezone.utc) if svc.started_at.tzinfo is None else svc.started_at
    delta = now - started
    days = delta.days
    hours = delta.seconds // 3600
    if days > 0:
        return f"{days}d {hours}h"
    return f"{hours}h"


# ── Response builder ─────────────────────────────────────────────────


def build_model_response(
    model: DeployedModel,
    svc: Optional[ServingService],
) -> DeployedModelResponse:
    """Construct a full API response from DB model + linked service."""
    status = _derive_status(svc)

    # Parse accelerators from serving service (e.g. "A100:4" -> type="A100", count=4)
    gpu_type = "\u2014"
    gpu_count = 0
    if svc and svc.accelerators:
        parts = svc.accelerators.split(":")
        gpu_type = parts[0]
        if len(parts) > 1:
            try:
                gpu_count = int(parts[1])
            except ValueError:
                gpu_count = 1
        else:
            gpu_count = 1

    return DeployedModelResponse(
        id=model.id,
        name=model.name,
        display_name=model.display_name,
        description="",
        base=model.base_model,
        family=model.family or "\u2014",
        param_count=model.param_count or "\u2014",
        type=model.model_type,
        quantization=model.quantization or "\u2014",
        context_window=model.context_window or 0,
        status=status,
        engine=model.engine or "\u2014",
        replicas=ReplicaInfo(
            current=svc.min_replicas if svc and status == "serving" else 0,
            desired=svc.min_replicas if svc else 0,
        ),
        gpu=GpuInfo(type=gpu_type, count=gpu_count, utilization=0),
        vram=VramInfo(used="0Gi", total="0Gi", pct=0),
        cpu="0%",
        mem="0Gi",
        mem_limit="\u2014",
        tps=0,
        p50="\u2014",
        p95="\u2014",
        p99="\u2014",
        queue_depth=0,
        batch_size="\u2014",
        tokens_in_24h="0",
        tokens_out_24h="0",
        requests_24h=0,
        error_rate="\u2014",
        uptime=_uptime(svc),
        last_deployed=_relative_time(model.last_deployed_at),
        deployed_by=model.deployed_by_id,
        namespace=model.namespace or "\u2014",
        project_color="#6B7280",
        endpoint=svc.endpoint or "\u2014" if svc else "\u2014",
        image=model.image or "\u2014",
        hub_ref=model.hub_ref or "",
        connected_agents=[],
        serving_config=model.serving_config,
        generation_defaults=model.generation_defaults,
        fine_tunes=[],
        metrics_history=MetricsHistoryInfo(),
        events=[],
    )


# ── Service functions ────────────────────────────────────────────────


async def deploy_model(
    session: AsyncSession,
    *,
    name: str,
    display_name: str,
    base_model: str,
    project_id: str,
    requested_by_id: str,
    family: Optional[str] = None,
    param_count: Optional[str] = None,
    model_type: str = "Base",
    quantization: Optional[str] = None,
    context_window: Optional[int] = None,
    engine: Optional[str] = None,
    image: Optional[str] = None,
    hub_ref: Optional[str] = None,
    namespace: Optional[str] = None,
    task_yaml: Optional[str] = None,
    accelerators: Optional[str] = None,
    cloud: Optional[str] = None,
    use_spot: bool = False,
    min_replicas: int = 1,
    max_replicas: Optional[int] = None,
    readiness_path: Optional[str] = None,
    load_balancing_policy: Optional[str] = None,
    serving_config: Optional[dict] = None,
    generation_defaults: Optional[dict] = None,
    server_config: Optional[ServerConfig] = None,
) -> DeployedModel:
    """Create a model record, resolving metadata from config files."""

    # Resolve model info from config.json / generation_config.json
    resolved: dict = {}
    try:
        if hub_ref and server_config:
            # LakeFS source — read config files from the repo
            client = await lakefs.get_lakefs_client(
                requested_by_id, session, server_config,
            )
            resolved = await resolve_from_lakefs(client, hub_ref)
        elif base_model and not hub_ref:
            # Hugging Face source — fetch config files via SDK
            resolved = await asyncio.to_thread(
                resolve_from_huggingface, base_model,
            )
    except Exception:
        logger.warning(
            f"Failed to resolve model info for {base_model}", exc_info=True,
        )

    # Merge resolved values — explicit request params take precedence
    if not family and resolved.get("family"):
        family = resolved["family"]
    if not param_count and resolved.get("param_count"):
        param_count = resolved["param_count"]
    if not quantization and resolved.get("quantization"):
        quantization = resolved["quantization"]
    if not context_window and resolved.get("context_window"):
        context_window = resolved["context_window"]
    if not generation_defaults and resolved.get("generation_defaults"):
        generation_defaults = resolved["generation_defaults"]

    svc_id: Optional[str] = None
    deployed_at: Optional[datetime] = None

    if task_yaml:
        try:
            svc = await sky_service.launch_serving_service(
                session,
                task_yaml=task_yaml,
                service_name=name,
                project_id=project_id,
                requested_by_id=requested_by_id,
                accelerators=accelerators,
                cloud=cloud,
                use_spot=use_spot,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                readiness_path=readiness_path,
                load_balancing_policy=load_balancing_policy,
            )
            svc_id = svc.id
            deployed_at = datetime.now(timezone.utc)
        except Exception:
            logger.warning(f"Failed to launch serving service for model {name}", exc_info=True)

    model = await repo.create_deployed_model(
        session,
        name=name,
        display_name=display_name,
        base_model=base_model,
        project_id=project_id,
        deployed_by_id=requested_by_id,
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
        serving_service_id=svc_id,
        last_deployed_at=deployed_at,
    )
    return model


async def update_model_config(
    session: AsyncSession,
    model_id: str,
    **values: object,
) -> Optional[DeployedModelResponse]:
    """Update serving configuration fields on a deployed model."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        return None

    await repo.update_deployed_model(session, model_id, **values)

    m = await repo.get_deployed_model(session, model_id)
    svc = None
    if m.serving_service_id:
        svc = await repo.get_serving_service(session, m.serving_service_id)
    return build_model_response(m, svc)


async def list_models(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """List deployed models with derived status from linked ServingServices."""
    db_models = await repo.list_deployed_models(
        session, project_id=project_id, search=search, limit=limit
    )

    results: list[DeployedModelResponse] = []
    status_counts: dict[str, int] = {
        "serving": 0, "deploying": 0, "error": 0, "stopped": 0,
    }

    for m in db_models:
        svc = None
        if m.serving_service_id:
            svc = await repo.get_serving_service(session, m.serving_service_id)

        resp = build_model_response(m, svc)
        status_counts[resp.status] = status_counts.get(resp.status, 0) + 1

        if status_filter and resp.status != status_filter:
            continue
        results.append(resp)

    return {
        "models": results,
        "total": len(results),
        "status_counts": status_counts,
    }


async def get_model(
    session: AsyncSession, model_id: str
) -> Optional[DeployedModelResponse]:
    """Get a single deployed model with full response."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        return None

    svc = None
    if m.serving_service_id:
        svc = await repo.get_serving_service(session, m.serving_service_id)

    return build_model_response(m, svc)


async def scale_model(
    session: AsyncSession,
    model_id: str,
    *,
    min_replicas: Optional[int] = None,
    max_replicas: Optional[int] = None,
) -> Optional[DeployedModelResponse]:
    """Scale a deployed model's replicas."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        return None
    if not m.serving_service_id:
        raise ValueError("Model is not currently serving")

    updates: dict = {}
    if min_replicas is not None:
        updates["min_replicas"] = min_replicas
    if max_replicas is not None:
        updates["max_replicas"] = max_replicas

    if updates:
        await repo.update_serving_service(session, m.serving_service_id, **updates)
        logger.info(f"Scaled model {model_id} replicas: {updates} (SkyPilot scaling not yet wired)")

    svc = await repo.get_serving_service(session, m.serving_service_id)
    return build_model_response(m, svc)


async def stop_model(session: AsyncSession, model_id: str) -> None:
    """Stop serving a model by terminating its linked ServingService."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        raise ValueError(f"Model {model_id} not found")
    if not m.serving_service_id:
        raise ValueError("Model is not currently serving")

    await sky_service.terminate_serving_service(session, m.serving_service_id)


async def restart_model(
    session: AsyncSession, model_id: str
) -> Optional[DeployedModelResponse]:
    """Restart a model by re-launching its ServingService."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        raise ValueError(f"Model {model_id} not found")

    # Get task_yaml from the old service
    task_yaml = None
    old_svc = None
    if m.serving_service_id:
        old_svc = await repo.get_serving_service(session, m.serving_service_id)
        if old_svc:
            task_yaml = old_svc.task_yaml

    if not task_yaml:
        raise ValueError("No task configuration available for restart")

    # Terminate old service if still active
    if old_svc and old_svc.status not in ("shutting_down", "failed", "failed_cleanup"):
        try:
            await sky_service.terminate_serving_service(session, m.serving_service_id)
        except Exception:
            logger.warning(f"Failed to terminate old service for model {model_id}", exc_info=True)

    # Launch new service
    svc = await sky_service.launch_serving_service(
        session,
        task_yaml=task_yaml,
        service_name=f"{m.name}-{int(datetime.now(timezone.utc).timestamp())}",
        project_id=m.project_id,
        requested_by_id=m.deployed_by_id,
        accelerators=old_svc.accelerators if old_svc else None,
        cloud=old_svc.cloud if old_svc else None,
        use_spot=old_svc.use_spot if old_svc else False,
        min_replicas=old_svc.min_replicas if old_svc else 1,
        max_replicas=old_svc.max_replicas if old_svc else None,
        readiness_path=old_svc.readiness_path if old_svc else None,
        load_balancing_policy=old_svc.load_balancing_policy if old_svc else None,
    )

    await repo.update_deployed_model(
        session, model_id,
        serving_service_id=svc.id,
        last_deployed_at=datetime.now(timezone.utc),
    )

    # Re-read the model to get updated state
    m = await repo.get_deployed_model(session, model_id)
    return build_model_response(m, svc)


async def delete_model(session: AsyncSession, model_id: str) -> None:
    """Terminate serving and delete the model record."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        raise ValueError(f"Model {model_id} not found")

    if m.serving_service_id:
        svc = await repo.get_serving_service(session, m.serving_service_id)
        if svc and svc.status not in ("shutting_down", "failed", "failed_cleanup"):
            try:
                await sky_service.terminate_serving_service(session, m.serving_service_id)
            except Exception:
                logger.warning(f"Failed to terminate service for model {model_id}", exc_info=True)

    await repo.delete_deployed_model(session, model_id)
