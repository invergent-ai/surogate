"""Pydantic models for agent/skill API routes."""

from typing import Optional

from pydantic import BaseModel


# ── Skill Requests ───────────────────────────────────────────────────


class SkillCreateRequest(BaseModel):
    name: str
    display_name: str
    description: str = ""
    content: str = ""
    tags: list[str] = []


class SkillUpdateRequest(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[list[str]] = None


# ── Skill Responses ──────────────────────────────────────────────────


class SkillResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: str
    content: str
    version: str
    status: str
    author_id: str
    author_username: str = ""
    tags: list[str] = []
    hub_ref: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    model_config = {"from_attributes": True}


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]
    total: int


class SkillPublishRequest(BaseModel):
    tag: str


class SkillPublishResponse(BaseModel):
    tag: str
    skill_id: str
    repository: str
