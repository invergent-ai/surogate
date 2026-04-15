"""Pydantic models for Agent API routes."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ── Agent Requests ──────────────────────────────────────────────────


class AgentCreateRequest(BaseModel):
    name: str
    harness: str
    display_name: str
    description: str = ""
    version: str = "0.1.0"
    model_id: Optional[str] = None
    status: str = "stopped"
    replicas: Optional[dict[str, Any]] = None
    image: str = ""
    endpoint: str = ""
    system_prompt: str = ""
    env_vars: Optional[dict[str, Any]] = None
    resources: Optional[dict[str, Any]] = None


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    harness: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    model_id: Optional[str] = None
    status: Optional[str] = None
    replicas: Optional[dict[str, Any]] = None
    image: Optional[str] = None
    endpoint: Optional[str] = None
    system_prompt: Optional[str] = None
    env_vars: Optional[dict[str, Any]] = None
    resources: Optional[dict[str, Any]] = None


# ── Agent Responses ─────────────────────────────────────────────────


class AgentResponse(BaseModel):
    id: str
    project_id: str
    project_name: str = ""
    name: str
    harness: str
    display_name: str
    description: str
    version: str
    model_id: Optional[str] = None
    model_name: str = ""
    status: str
    replicas: Optional[dict[str, Any]] = None
    image: str
    endpoint: str
    system_prompt: str = ""
    env_vars: Optional[dict[str, Any]] = None
    resources: Optional[dict[str, Any]] = None
    created_by_id: str
    created_by_username: str = ""
    hub_ref: Optional[str] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AgentListResponse(BaseModel):
    agents: list[AgentResponse]
    total: int
