"""Pydantic models for local task API routes."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class TaskSpawnRequest(BaseModel):
    task_type: str  # import_model / import_dataset / local_inference / local_training
    name: str
    project_id: str
    params: dict[str, Any]  # task-specific: hf_repo_id, lakefs_repo_id, etc.


class LocalTaskResponse(BaseModel):
    id: str
    name: str
    task_type: str
    status: str
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    progress: Optional[str] = None
    project_id: str
    requested_by: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TaskLogsResponse(BaseModel):
    task_id: str
    lines: list[str]
