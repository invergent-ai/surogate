"""Deployed-model API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute import models as models_service
from surogate.core.db.engine import get_session
from surogate.server.auth.authentication import get_current_subject, get_current_user_id
from surogate.server.models.models import (
    DeployedModelCreateRequest,
    DeployedModelListResponse,
    DeployedModelResponse,
    DeployedModelScaleRequest,
    DeployedModelUpdateRequest,
    ModelEventsResponse,
    ModelLogsResponse,
)

router = APIRouter()


@router.get("/", response_model=DeployedModelListResponse)
async def list_models(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """List deployed models with optional filters."""
    return await models_service.list_models(
        session,
        project_id=project_id,
        status_filter=status,
        search=search,
        limit=limit,
    )


@router.get("/{model_id}", response_model=DeployedModelResponse)
async def get_model(
    model_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Get a single deployed model."""
    resp = await models_service.get_model(session, model_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return resp


@router.post("/", response_model=DeployedModelResponse)
async def create_model(
    req: DeployedModelCreateRequest,
    request: Request,
    current_user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    """Create a new model record, resolving metadata from config files."""
    model = await models_service.create_model(
        session,
        name=req.name,
        display_name=req.display_name,
        base_model=req.base_model,
        project_id=req.project_id,
        requested_by_id=current_user_id,
        family=req.family,
        param_count=req.param_count,
        model_type=req.model_type,
        quantization=req.quantization,
        context_window=req.context_window,
        engine=req.engine,
        image=req.image,
        hub_ref=req.hub_ref,
        source=req.source,
        serving_config=req.serving_config,
        generation_defaults=req.generation_defaults,
        server_config=request.app.state.config,
    )
    resp = await models_service.get_model(session, model.id)
    return resp


@router.patch("/{model_id}", response_model=DeployedModelResponse)
async def update_model(
    model_id: str,
    req: DeployedModelUpdateRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Update a deployed model's configuration."""
    resp = await models_service.update_model_config(
        session, model_id,
        engine=req.engine,
        serving_config=req.serving_config,
        generation_defaults=req.generation_defaults,
        infra=req.infra,
        instance_type=req.instance_type,
        region=req.region,
        accelerators=req.accelerators,
        use_spot=req.use_spot,
    )
    if resp is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return resp


@router.post("/{model_id}/scale", response_model=DeployedModelResponse)
async def scale_model(
    model_id: str,
    req: DeployedModelScaleRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Scale a deployed model's replicas."""
    try:
        resp = await models_service.scale_model(
            session, model_id,
            replicas=req.replicas,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if resp is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return resp


@router.post("/{model_id}/start", response_model=DeployedModelResponse)
async def start_model(
    model_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Start serving a stopped model via dstack."""
    try:
        resp = await models_service.start_model(
            session, model_id, server_config=request.app.state.config
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return resp


@router.post("/{model_id}/restart", response_model=DeployedModelResponse)
async def restart_model(
    model_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Restart a deployed model."""
    try:
        resp = await models_service.restart_model(
            session, model_id, server_config=request.app.state.config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return resp


@router.post("/{model_id}/stop")
async def stop_model(
    model_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Stop serving a deployed model."""
    try:
        await models_service.stop_model(session, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "stopping"}


@router.get("/{model_id}/logs", response_model=ModelLogsResponse)
async def get_model_logs(
    model_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    tail: int = Query(200, le=1000),
):
    """Get the last N lines of logs for a model's serving service."""
    try:
        return await models_service.get_model_logs(
            session, model_id, tail=tail,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{model_id}/events", response_model=ModelEventsResponse)
async def get_model_events(
    model_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(100, le=500),
):
    """Get dstack events for a model's serving service."""
    try:
        return await models_service.get_model_events(
            session, model_id, limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Delete a deployed model."""
    try:
        await models_service.delete_model(session, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "deleted"}
