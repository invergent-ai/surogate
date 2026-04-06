"""Local task API routes — spawn, list, cancel, logs."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

import sqlalchemy as sa
from surogate.core.db.engine import get_session
from surogate.core.db.models.compute import LocalTask, LocalTaskStatus, LocalTaskType
from surogate.server.auth.authentication import get_current_subject, get_current_user_id
from surogate.server.models.tasks import (
    LocalTaskResponse,
    TaskLogsResponse,
    TaskSpawnRequest,
)

router = APIRouter()


def _to_response(t: LocalTask) -> LocalTaskResponse:
    return LocalTaskResponse(
        id=t.id,
        name=t.name,
        task_type=t.task_type.value,
        status=t.status.value,
        pid=t.pid,
        exit_code=t.exit_code,
        error_message=t.error_message,
        progress=t.progress,
        project_id=t.project_id,
        requested_by=t.requested_by_id,
        created_at=t.created_at,
        started_at=t.started_at,
        completed_at=t.completed_at,
    )


@router.get("", response_model=list[LocalTaskResponse])
async def list_tasks(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """List local tasks with optional filters."""
    stmt = sa.select(LocalTask).order_by(LocalTask.created_at.desc())
    if project_id:
        stmt = stmt.where(LocalTask.project_id == project_id)
    if task_type:
        stmt = stmt.where(LocalTask.task_type == LocalTaskType(task_type))
    if status:
        stmt = stmt.where(LocalTask.status == LocalTaskStatus(status))
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return [_to_response(t) for t in result.scalars().all()]


@router.post("", response_model=LocalTaskResponse)
async def spawn_task(
    req: TaskSpawnRequest,
    request: Request,
    current_user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    """Spawn a new local task as a subprocess."""
    manager = request.app.state.task_manager
    task = await manager.spawn(
        session,
        task_type=req.task_type,
        name=req.name,
        project_id=req.project_id,
        user_id=current_user_id,
        params=req.params,
    )
    return _to_response(task)


@router.get("/{task_id}", response_model=LocalTaskResponse)
async def get_task(
    task_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Get a single task's details."""
    result = await session.execute(
        sa.select(LocalTask).where(LocalTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _to_response(task)


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Cancel a running task via SIGTERM."""
    manager = request.app.state.task_manager
    try:
        await manager.cancel(session, task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "cancelled"}


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Cancel (if running) and delete a task record."""
    result = await session.execute(
        sa.select(LocalTask).where(LocalTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in (LocalTaskStatus.pending, LocalTaskStatus.running):
        manager = request.app.state.task_manager
        try:
            await manager.cancel(session, task_id)
        except ValueError:
            pass
    await session.delete(task)
    await session.commit()
    return {"status": "deleted"}


@router.get("/{task_id}/logs", response_model=TaskLogsResponse)
async def get_task_logs(
    task_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    tail: int = Query(100, le=1000),
):
    """Get the last N lines from a task's log file."""
    manager = request.app.state.task_manager
    lines = await manager.get_logs(task_id, tail=tail)
    return TaskLogsResponse(task_id=task_id, lines=lines)
