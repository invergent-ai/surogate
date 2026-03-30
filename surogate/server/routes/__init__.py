"""
API Routes
"""
from routes.auth import router as auth_router
from routes.project import router as project_router
from routes.lakefs import router as hub_router
from routes.compute import router as compute_router
from routes.tasks import router as tasks_router

__all__ = [
    "auth_router",
    "project_router",
    "hub_router",
    "compute_router",
    "tasks_router",
]
