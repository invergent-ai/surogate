"""Pydantic models for project API routes."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


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

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v):
        return v.value if hasattr(v, "value") else str(v)

    model_config = ConfigDict(from_attributes=True)
