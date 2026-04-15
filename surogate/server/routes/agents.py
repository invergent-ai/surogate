"""Agent CRUD API routes."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.compute.agent_monitor import agent_view
from surogate.core.db.engine import get_session
from surogate.core.db.models.operate import AgentStatus
from surogate.core.db.repository import agents as agent_repo
from surogate.core.db.repository import api_keys as api_keys_repo
from surogate.core.db.repository.api_keys import agent_scope, model_scope
from surogate.core.db.repository import user as user_repo
from surogate.server.auth.authentication import get_current_subject
from surogate.server.models.agent import (
    AgentCreateRequest,
    AgentListResponse,
    AgentResponse,
    AgentUpdateRequest,
)
from surogate.utils.logger import get_logger

logger = get_logger()


class AgentUserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    auth_provider: str
    created_at: Optional[str] = None


class AgentUserListResponse(BaseModel):
    users: list[AgentUserResponse]
    total: int


class AgentUserCreateRequest(BaseModel):
    email: str
    display_name: str
    password: Optional[str] = None


class AgentUserUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    password: Optional[str] = None


def _user_to_response(u) -> AgentUserResponse:
    return AgentUserResponse(
        id=str(u.id),
        email=u.email,
        display_name=u.display_name,
        auth_provider=u.auth_provider,
        created_at=u.created_at.isoformat() if u.created_at else None,
    )

router = APIRouter()

# ── helpers ──────────────────────────────────────────────────────────


def _agent_to_response(agent, agent_base_domain: str = "") -> AgentResponse:
    return AgentResponse(**agent_view(agent, agent_base_domain))


def _domain(request: Request) -> str:
    return getattr(request.app.state.config, "agent_base_domain", "")


# ── Agent CRUD ───────────────────────────────────────────────────────


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    harness: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
):
    agents = await agent_repo.list_agents(
        session,
        project_id=project_id,
        status=status,
        harness=harness,
        limit=limit,
    )
    return AgentListResponse(
        agents=[_agent_to_response(a, _domain(request)) for a in agents],
        total=len(agents),
    )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    agent = await agent_repo.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_response(agent, _domain(request))


@router.post("/agents", response_model=AgentResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: str = Query(...),
):
    user = await user_repo.get_user_by_username(session, current_subject)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    try:
        agent_status = AgentStatus(body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")

    agent = await agent_repo.create_agent(
        session,
        project_id=project_id,
        name=body.name,
        harness=body.harness,
        display_name=body.display_name,
        description=body.description,
        version=body.version,
        model_id=body.model_id,
        status=agent_status,
        replicas=body.replicas,
        image=body.image,
        endpoint=body.endpoint,
        system_prompt=body.system_prompt,
        env_vars=body.env_vars,
        resources=body.resources,
        created_by_id=user.id,
    )
    return _agent_to_response(agent, _domain(request))


@router.patch("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    body: AgentUpdateRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await agent_repo.get_agent(session, agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    fields = body.model_dump(exclude_unset=True)
    if not fields:
        return _agent_to_response(existing, _domain(request))

    agent = await agent_repo.update_agent(session, agent_id, **fields)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_response(agent, _domain(request))


@router.delete("/agents/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await agent_repo.get_agent(session, agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    from surogate.core.compute.helm import stop_agent as helm_stop
    await helm_stop(existing, existing.project, request.app.state.config)
    await api_keys_repo.delete_by_scope(session, agent_scope(existing.id))
    try:
        await request.app.state.surogates.delete_agent_data(existing.id)
    except Exception as exc:
        logger.warning(
            "Failed to scrub surogates data for agent %s: %r",
            existing.id, exc, exc_info=True,
        )
    await agent_repo.delete_agent(session, agent_id)


@router.post("/agents/{agent_id}/start", response_model=AgentResponse)
async def start_agent(
    agent_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    agent = await agent_repo.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.project is None:
        raise HTTPException(status_code=400, detail="Agent has no project")

    from surogate.core.compute.helm import start_agent as helm_start
    from uuid import UUID

    if agent.model is None:
        raise HTTPException(status_code=400, detail="Agent has no model configured")

    model = agent.model
    config = request.app.state.config

    surogates = request.app.state.surogates
    await surogates.ensure_org(UUID(agent.project.id), agent.project.name)

    user = await user_repo.get_user_by_username(session, current_subject)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    await api_keys_repo.delete_by_scope(session, agent_scope(agent.id))
    _key_row, raw_key = await api_keys_repo.create_api_key(
        session,
        name=agent_scope(agent.name),
        scopes=[agent_scope(agent.id), model_scope(model.id)],
        created_by_id=user.id,
    )

    model_name = (
        model.base_model
        if model.engine in ("openrouter", "openai_compat")
        else model.name
    )
    llm = {
        "model": model_name,
        "baseUrl": (
            f"{config.platform_api_url.rstrip('/')}"
            f"/proxy/services/_model/{model.id}/v1"
        ),
        "apiKey": raw_key,
    }

    await agent_repo.update_agent(session, agent_id, status=AgentStatus.deploying)
    try:
        await helm_start(agent, agent.project, config, llm, org_id=agent.project.id)
    except Exception as exc:
        await agent_repo.update_agent(session, agent_id, status=AgentStatus.error)
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {exc}")

    updated = await agent_repo.get_agent(session, agent_id)
    return _agent_to_response(updated, _domain(request))


@router.post("/agents/{agent_id}/stop", response_model=AgentResponse)
async def stop_agent(
    agent_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    agent = await agent_repo.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.project is None:
        raise HTTPException(status_code=400, detail="Agent has no project")

    from surogate.core.compute.helm import stop_agent as helm_stop

    try:
        await helm_stop(agent, agent.project, request.app.state.config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop agent: {exc}")

    await api_keys_repo.revoke_by_scope(session, agent_scope(agent.id))
    await agent_repo.update_agent(session, agent_id, status=AgentStatus.stopped)
    updated = await agent_repo.get_agent(session, agent_id)
    return _agent_to_response(updated, _domain(request))


# ── Users (surogates-db tenant users, scoped to the agent's org) ─────


async def _resolve_agent_org(
    agent_id: str, session: AsyncSession, surogates,
) -> UUID:
    """Load the agent, ensure the surogates org exists, return its UUID."""
    agent = await agent_repo.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.project is None:
        raise HTTPException(status_code=400, detail="Agent has no project")
    org_id = UUID(agent.project.id)
    # Ensure the org row exists (idempotent) — lets the UI manage users
    # for an agent that has not been started yet.
    await surogates.ensure_org(org_id, agent.project.name)
    return org_id


@router.get("/agents/{agent_id}/users", response_model=AgentUserListResponse)
async def list_agent_users(
    agent_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    surogates = request.app.state.surogates
    org_id = await _resolve_agent_org(agent_id, session, surogates)
    users = await surogates.list_users(org_id)
    return AgentUserListResponse(
        users=[_user_to_response(u) for u in users],
        total=len(users),
    )


@router.post(
    "/agents/{agent_id}/users",
    response_model=AgentUserResponse,
    status_code=201,
)
async def create_agent_user(
    agent_id: str,
    body: AgentUserCreateRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    surogates = request.app.state.surogates
    org_id = await _resolve_agent_org(agent_id, session, surogates)
    try:
        user = await surogates.create_user(
            org_id=org_id,
            email=body.email,
            display_name=body.display_name,
            password=body.password,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create user: {exc}")
    return _user_to_response(user)


@router.patch(
    "/agents/{agent_id}/users/{user_id}",
    response_model=AgentUserResponse,
)
async def update_agent_user(
    agent_id: str,
    user_id: str,
    body: AgentUserUpdateRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    surogates = request.app.state.surogates
    await _resolve_agent_org(agent_id, session, surogates)
    user = await surogates.update_user(
        UUID(user_id),
        display_name=body.display_name,
        password=body.password,
    )
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_response(user)


@router.delete(
    "/agents/{agent_id}/users/{user_id}",
    status_code=204,
)
async def delete_agent_user(
    agent_id: str,
    user_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    surogates = request.app.state.surogates
    await _resolve_agent_org(agent_id, session, surogates)
    ok = await surogates.delete_user(UUID(user_id))
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
