"""Service layer for deployed models.

Bridges DeployedModel (model identity + config) with ServingService
(infrastructure / SkyPilot).  Status is derived from the linked
ServingService state rather than stored on the model itself.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import yaml

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



def _derive_status(svc: Optional[ServingService]) -> str:
    if svc is None:
        return "stopped"
    return svc.status


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
        project_id=model.project_id,
        family=model.family or "\u2014",
        param_count=model.param_count or "\u2014",
        type=model.model_type,
        quantization=model.quantization or "\u2014",
        context_window=model.context_window or 0,
        status=status,
        engine=model.engine or "\u2014",
        replicas=ReplicaInfo(
            current=svc.replicas if svc and status == "serving" else 0,
            desired=svc.replicas if svc else 0,
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


async def _unique_service_name(session: AsyncSession, base: str) -> str:
    """Return *base* if unused, otherwise append ``-2``, ``-3``, etc."""
    existing = await repo.get_serving_service_by_name(session, base)
    if existing is None:
        return base
    n = 2
    while True:
        candidate = f"{base}-{n}"
        if await repo.get_serving_service_by_name(session, candidate) is None:
            return candidate
        n += 1


async def create_model(
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
    serving_config: Optional[dict] = None,
    generation_defaults: Optional[dict] = None,
    server_config: Optional[ServerConfig] = None,
) -> DeployedModel:
    """Create a model record, resolving metadata from config files.

    This only creates the DB record — call start_model() to actually
    launch the serving service via SkyPilot.
    """

    # Resolve model info from config.json / generation_config.json
    resolved: dict = {}
    try:
        if hub_ref and server_config:
            client = await lakefs.get_lakefs_client(
                requested_by_id, session, server_config,
            )
            resolved = await resolve_from_lakefs(client, hub_ref)
        elif base_model and not hub_ref:
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

    # Build service name: <project_namespace>-<model_slug>, deduplicated
    import sqlalchemy as sa
    from surogate.core.db.models.platform import Project
    proj = (await session.execute(
        sa.select(Project.namespace).where(Project.id == project_id)
    )).scalar_one_or_none()
    svc_base = f"{proj}-{name}" if proj else name
    svc_name = await _unique_service_name(session, svc_base)

    # Create the ServingService first (defaults to k8s, stopped)
    svc = await repo.create_serving_service(
        session,
        name=svc_name,
        project_id=project_id,
        requested_by_id=requested_by_id,
        task_yaml="",
        infra="k8s",
        status="stopped",
    )

    model = await repo.create_deployed_model(
        session,
        name=svc_name,
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
        serving_service_id=svc.id,
    )
    return model


async def update_model_config(
    session: AsyncSession,
    model_id: str,
    *,
    engine: Optional[str] = None,
    serving_config: Optional[dict] = None,
    generation_defaults: Optional[dict] = None,
    infra: Optional[str] = None,
    accelerators: Optional[str] = None,
    use_spot: Optional[bool] = None,
) -> Optional[DeployedModelResponse]:
    """Update serving configuration fields on a deployed model."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        return None

    # Model-level fields -> DeployedModel
    model_updates: dict = {}
    if engine is not None:
        model_updates["engine"] = engine
    if serving_config is not None:
        model_updates["serving_config"] = serving_config
    if generation_defaults is not None:
        model_updates["generation_defaults"] = generation_defaults
    if model_updates:
        await repo.update_deployed_model(session, model_id, **model_updates)

    # Infra fields -> ServingService
    svc_updates: dict = {}
    if infra is not None:
        svc_updates["infra"] = infra
    if accelerators is not None:
        svc_updates["accelerators"] = accelerators
    if use_spot is not None:
        svc_updates["use_spot"] = use_spot
    if svc_updates:
        if m.serving_service_id:
            await repo.update_serving_service(session, m.serving_service_id, **svc_updates)
        else:
            svc = await repo.create_serving_service(
                session,
                name=m.name,
                project_id=m.project_id,
                requested_by_id=m.deployed_by_id,
                task_yaml="",
                status="stopped",
                **svc_updates,
            )
            await repo.update_deployed_model(
                session, model_id, serving_service_id=svc.id,
            )

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
    status_counts: dict[str, int] = {}

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
    replicas: int,
) -> Optional[DeployedModelResponse]:
    """Scale a deployed model's replicas."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        return None
    if not m.serving_service_id:
        raise ValueError("Model is not currently serving")

    await repo.update_serving_service(
        session, m.serving_service_id, replicas=replicas,
    )
    logger.info(f"Scaled model {model_id} to {replicas} replicas (SkyPilot scaling not yet wired)")

    svc = await repo.get_serving_service(session, m.serving_service_id)
    return build_model_response(m, svc)


_MODEL_DIR = "/models"


def _build_task_yaml(
    model: DeployedModel,
    svc: ServingService,
    *,
    lakefs_s3_endpoint: Optional[str] = None,
    lakefs_k8s_s3_endpoint: Optional[str] = None,
    lakefs_access_key: Optional[str] = None,
    lakefs_secret_key: Optional[str] = None,
) -> str:
    """Generate a SkyPilot serving task YAML for llama.cpp."""

    hub = _parse_hub_ref(model.hub_ref) if model.hub_ref else None

    # ── Run command ─────────────────────────────────────────────────
    if hub:
        # Download model from LakeFS via rclone, then start server.
        # Auto-detect: use the GGUF file if one exists, else directory.
        run_cmd = (
            "echo \"Downloading model from ${RCLONE_CONFIG_LAKEFS_ENDPOINT}...\" && "
            f"rclone copy --no-check-certificate lakefs:{hub['repo']}/{hub['branch']}/ {_MODEL_DIR} && "
            f"GGUF=$(find {_MODEL_DIR} -maxdepth 1 -name '*.gguf' | head -1) && "
            f"MODEL_PATH=${{GGUF:-{_MODEL_DIR}}} && "
            "cd /app && ./llama-server --metrics --host 0.0.0.0 --port 8080"
            " -m $MODEL_PATH"
        )
    else:
        run_cmd = (
            "cd /app && ./llama-server --metrics --host 0.0.0.0 --port 8080"
            f" -m {model.base_model}"
        )

    # TODO: add --lora for serving a lora adapter, hf token and sampling params
    if model.context_window and model.context_window > 0:
        run_cmd += f" -c {model.context_window}"

    resources: dict = {
        "image_id": "docker:ghcr.io/invergent-ai/sky-llama-cpp:full-cuda12",
        "ports": 8080,
        "infra": svc.infra or "k8s",
        "use_spot": svc.use_spot or False,
        "accelerators": "RTX4070-LAPTOP-GPU:1",
    }
    if svc.accelerators:
        resources["accelerators"] = svc.accelerators

    task: dict = {
        "resources": resources,
        "secrets": {
            "HF_TOKEN": ""
        },
        "service": {
            "readiness_probe": svc.readiness_path or "/health",
            "replicas": svc.replicas,
            "ports": 8080,
        },
        "run": run_cmd,
    }

    # Pass rclone config for LakeFS S3 gateway via env vars — same
    # pattern as local_tasks.py uses for subprocess-based imports.
    if hub and lakefs_access_key and lakefs_secret_key:
        task["envs"] = {
            "RCLONE_CONFIG_LAKEFS_TYPE": "s3",
            "RCLONE_CONFIG_LAKEFS_PROVIDER": "Other",
            "RCLONE_CONFIG_LAKEFS_ENV_AUTH": "false",
            "RCLONE_CONFIG_LAKEFS_NO_CHECK_BUCKET": "true",
            "RCLONE_CONFIG_LAKEFS_ENDPOINT": lakefs_k8s_s3_endpoint if svc.infra == "k8s" else lakefs_s3_endpoint,
            "RCLONE_CONFIG_LAKEFS_ACCESS_KEY_ID": lakefs_access_key,
            "RCLONE_CONFIG_LAKEFS_SECRET_ACCESS_KEY": lakefs_secret_key,
        }

    if svc.load_balancing_policy:
        task["service"]["load_balancing_policy"] = svc.load_balancing_policy

    return yaml.dump(task, default_flow_style=False)


def _parse_hub_ref(hub_ref: str) -> Optional[dict]:
    """Parse ``repo@branch`` into ``{'repo': ..., 'branch': ...}``."""
    parts = hub_ref.split("@", 1)
    if len(parts) != 2:
        return None
    return {"repo": parts[0], "branch": parts[1]}


async def start_model(
    session: AsyncSession,
    model_id: str,
    server_config: Optional[ServerConfig] = None,
) -> Optional[DeployedModelResponse]:
    """Start serving a stopped model by generating a task YAML and launching via SkyPilot."""
    m = await repo.get_deployed_model(session, model_id)
    
    if m is None:
        raise ValueError(f"Model {model_id} not found")

    if not m.serving_service_id:
        raise ValueError(f"No serving service configured for model {model_id}")

    svc = await repo.get_serving_service(session, m.serving_service_id)
    if svc is None:
        raise ValueError(f"No serving service configured for model {model_id}")

    # Fetch per-user LakeFS credentials for hub models
    lakefs_kw: dict = {}
    if m.hub_ref and server_config and server_config.lakefs_s3_endpoint:
        import surogate.core.db.repository.user as user_repo
        creds = await user_repo.get_lakefs_credentials(session, m.deployed_by_id)
        if creds and all(creds):
            lakefs_kw = {
                "lakefs_s3_endpoint": server_config.lakefs_s3_endpoint,
                "lakefs_k8s_s3_endpoint": server_config.lakefs_k8s_s3_endpoint,
                "lakefs_access_key": creds[0],
                "lakefs_secret_key": creds[1],
            }

    # Terminate if still running
    if svc.status not in ("shutting_down", "failed", "failed_cleanup", "stopped"):
        try:
            await sky_service.terminate_serving_service(session, m.serving_service_id)
        except Exception:
            logger.warning(f"Failed to terminate old service for model {model_id}", exc_info=True)

    task_yaml = _build_task_yaml(m, svc, **lakefs_kw)
    
    svc = await sky_service.launch_serving_service(session, svc, task_yaml)

    await repo.update_deployed_model(
        session, model_id,
        last_deployed_at=datetime.now(timezone.utc),
    )

    m = await repo.get_deployed_model(session, model_id)
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
    session: AsyncSession,
    model_id: str,
    server_config: Optional[ServerConfig] = None,
) -> Optional[DeployedModelResponse]:
    """Restart a model by stopping then starting it."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        raise ValueError(f"Model {model_id} not found")

    if m.serving_service_id:
        try:
            await stop_model(session, model_id)
        except ValueError:
            pass  # already stopped

    return await start_model(session, model_id, server_config=server_config)


async def get_model_logs(
    session: AsyncSession,
    model_id: str,
    *,
    target: str = "controller",
    replica_id: Optional[int] = None,
    tail: int = 200,
) -> dict:
    """Fetch logs for a deployed model's serving service."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        raise ValueError(f"Model {model_id} not found")
    if not m.serving_service_id:
        raise ValueError("Model has no serving service")

    svc = await repo.get_serving_service(session, m.serving_service_id)
    if svc is None:
        raise ValueError("Serving service not found")

    lines = await sky_service.tail_serving_logs(
        svc.name, target=target, replica_id=replica_id, tail=tail,
    )
    return {"model_id": model_id, "target": target, "lines": lines}


async def delete_model(session: AsyncSession, model_id: str) -> None:
    """Delete the model and its ServingService, tearing down SkyPilot in the background."""
    m = await repo.get_deployed_model(session, model_id)
    if m is None:
        raise ValueError(f"Model {model_id} not found")

    serving_service_id = m.serving_service_id
    service_name: str | None = None
    if serving_service_id:
        svc = await repo.get_serving_service(session, serving_service_id)
        if svc is not None:
            service_name = svc.name

    # Delete DB records immediately so the UI can move on
    await repo.delete_deployed_model(session, model_id)
    if serving_service_id:
        await repo.delete_serving_service(session, serving_service_id)

    # Tear down the SkyPilot service in the background
    if service_name:
        asyncio.create_task(_teardown_sky_service(service_name))


async def _teardown_sky_service(service_name: str) -> None:
    """Best-effort SkyPilot service teardown — runs as a background task."""
    from sky.serve.server import core as serve_core

    try:
        await asyncio.to_thread(serve_core.down, service_names=service_name, purge=False)
        logger.info("SkyPilot service '%s' torn down", service_name)
    except Exception:
        logger.warning("Failed to tear down SkyPilot service '%s'", service_name, exc_info=True)
