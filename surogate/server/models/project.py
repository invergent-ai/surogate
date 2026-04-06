"""Pydantic models for project API routes."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, validator


class CreateProjectRequest(BaseModel):
    name: str
    color: str


class ProjectResponse(BaseModel):
    id: str
    name: str
    namespace: str
    color: str
    status: str
    created_by_id: str
    created_at: datetime
    dstack_project_name: Optional[str] = None

    @validator("status", pre=True)
    def coerce_status(cls, v):
        return v.value if hasattr(v, "value") else str(v)

    class Config:
        orm_mode = True
