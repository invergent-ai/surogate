"""
Project API routes
"""

import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.engine import get_session
from surogate.core.db.repository import project as project_repository
from surogate.server.auth.authentication import get_current_subject
from surogate.server.models.project import CreateProjectRequest, ProjectResponse

router = APIRouter()

@router.get("", response_model=list[ProjectResponse])
async def get_user_projects(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> list[ProjectResponse]:
    """Fetch the list of projects the user has access to."""
    projects = await project_repository.get_user_projects(session, current_subject)
    return [ProjectResponse.model_validate(p) for p in projects]


def _slugify(name: str) -> str:
    """Turn a project name into a URL-safe namespace slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "project"


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: CreateProjectRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> ProjectResponse:
    """Create a new project."""
    namespace = _slugify(body.name)
    try:
        project = await project_repository.create_project(
            session,
            name=body.name,
            namespace=namespace,
            color=body.color,
            username=current_subject,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProjectResponse.model_validate(project)
