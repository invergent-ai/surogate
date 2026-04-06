"""Pydantic models for agent/skill API routes."""

from typing import Optional

from pydantic import BaseModel


# ── Skill Requests ───────────────────────────────────────────────────


class SkillCreateRequest(BaseModel):
    name: str
    display_name: str
    description: str = ""
    content: str = ""
    license: str = ""
    compatibility: str = ""
    metadata: Optional[dict[str, str]] = None
    allowed_tools: list[str] = []
    status: str = "active"
    tags: list[str] = []


class SkillUpdateRequest(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    allowed_tools: Optional[list[str]] = None
    status: Optional[str] = None
    tags: Optional[list[str]] = None


# ── Skill Responses ──────────────────────────────────────────────────


class SkillResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: str
    content: str
    license: str
    compatibility: str = ""
    metadata: Optional[dict[str, str]] = None
    allowed_tools: list[str] = []
    status: str
    author_id: str
    author_username: str = ""
    tags: list[str] = []
    hub_ref: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        orm_mode = True


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]
    total: int


class SkillPublishRequest(BaseModel):
    tag: str


class SkillPublishResponse(BaseModel):
    tag: str
    skill_id: str
    repository: str
