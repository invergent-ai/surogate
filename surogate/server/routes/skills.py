"""Agent & Skill API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from lakefs_sdk import ApiException, RepositoryCreation
from sqlalchemy.ext.asyncio import AsyncSession

import surogate.core.hub.lakefs as lakefs
from surogate.core.db.engine import get_session
from surogate.core.db.models.operate import SkillStatus
from surogate.core.db.repository import skills as skill_repo
from surogate.core.db.repository import user as user_repo
from surogate.server.auth.authentication import get_current_subject
from surogate.server.models.skill import (
    SkillCreateRequest,
    SkillListResponse,
    SkillPublishRequest,
    SkillPublishResponse,
    SkillResponse,
    SkillUpdateRequest,
)

router = APIRouter()


# ── helpers ──────────────────────────────────────────────────────────


def _skill_to_response(skill) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        name=skill.name,
        display_name=skill.display_name,
        description=skill.description,
        content=skill.content,
        version=skill.version,
        status=skill.status.value if hasattr(skill.status, "value") else skill.status,
        author_id=skill.author_id,
        author_username=skill.author.username if skill.author else "",
        tags=skill.tags or [],
        hub_ref=skill.hub_ref,
        created_at=str(skill.created_at) if skill.created_at else None,
        updated_at=str(skill.updated_at) if skill.updated_at else None,
    )


# ── Skill CRUD ───────────────────────────────────────────────────────


@router.get("/skills", response_model=SkillListResponse)
async def list_skills(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
):
    skills = await skill_repo.list_skills(
        session, project_id=project_id, status=status, limit=limit,
    )
    return SkillListResponse(
        skills=[_skill_to_response(s) for s in skills],
        total=len(skills),
    )


@router.get("/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(
    skill_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    skill = await skill_repo.get_skill(session, skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_to_response(skill)


@router.post("/skills", response_model=SkillResponse, status_code=201)
async def create_skill(
    body: SkillCreateRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: str = Query(...),
):
    from surogate.core.db.models.platform import User
    import sqlalchemy as sa

    user = await user_repo.get_user_by_username(session, current_subject)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    try:
        skill_status = SkillStatus(body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")

    # Create a LakeFS repository for this skill
    config = request.app.state.config
    repo_name = f"skill-{body.name}"
    api_client = await lakefs.get_lakefs_client(current_subject, session, config)
    try:
        repo = await lakefs.create_repository(
            api_client,
            current_subject,
            RepositoryCreation(name=repo_name, storage_namespace=f"local://{repo_name}"),
            config,
        )
    except ApiException as e:
        if e.status == 409:
            raise HTTPException(status_code=409, detail=f"Repository '{repo_name}' already exists")
        raise HTTPException(status_code=500, detail=f"Failed to create skill repository '{repo_name}'")
    if repo is None:
        raise HTTPException(status_code=500, detail=f"Failed to create skill repository '{repo_name}'")

    skill = await skill_repo.create_skill(
        session,
        name=body.name,
        display_name=body.display_name,
        description=body.description,
        content=body.content,
        version=body.version,
        status=skill_status,
        project_id=project_id,
        author_id=user.id,
        tags=body.tags,
        hub_ref=repo.id,
    )
    return _skill_to_response(skill)


@router.patch("/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: str,
    body: SkillUpdateRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await skill_repo.get_skill(session, skill_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Skill not found")

    fields = body.model_dump(exclude_unset=True)
    if not fields:
        return _skill_to_response(existing)

    skill = await skill_repo.update_skill(session, skill_id, **fields)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_to_response(skill)


@router.post("/skills/{skill_id}/publish", response_model=SkillPublishResponse)
async def publish_skill(
    skill_id: str,
    body: SkillPublishRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await skill_repo.get_skill(session, skill_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Skill not found")

    config = request.app.state.config
    repo_name = f"skill-{existing.name}"
    api_client = await lakefs.get_lakefs_client(current_subject, session, config)

    tag_ref = await lakefs.create_tag(api_client, repo_name, body.tag, "main")
    if tag_ref is None:
        raise HTTPException(status_code=500, detail=f"Failed to create tag '{body.tag}'")

    # Update the skill version to match the published tag
    await skill_repo.update_skill(session, skill_id, version=body.tag)

    return SkillPublishResponse(
        tag=body.tag,
        skill_id=skill_id,
        repository=repo_name,
    )


@router.delete("/skills/{skill_id}", status_code=204)
async def delete_skill(
    skill_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await skill_repo.get_skill(session, skill_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Skill not found")

    # Delete the associated LakeFS repository
    config = request.app.state.config
    repo_name = f"skill-{existing.name}"
    api_client = await lakefs.get_lakefs_client(current_subject, session, config)
    await lakefs.delete_repository(api_client, repo_name, current_subject, config)

    await skill_repo.delete_skill(session, skill_id)
