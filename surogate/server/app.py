"""
Main FastAPI application for Surogate
"""
import asyncio
import os
import sys
from pathlib import Path as _Path
from surogate.core.config.server_config import ServerConfig
from surogate.core.db.engine import get_session_factory
from surogate.server.auth.authentication import get_current_subject
from surogate.utils.logger import get_logger
from surogate.server.notifier import manager as ws_manager, notify_transition

# Ensure backend dir is on sys.path so _platform_compat is importable when
# main.py is launched directly (e.g. `uvicorn main:app`).
_backend_dir = str(_Path(__file__).parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pathlib import Path

# the order of imports is important here. init_dstack() must be called before any dstack imports,
# and it must be called in the main thread to properly set up context propagation for threads.
from surogate.core.compute import init_dstack, shutdown_dstack
from surogate.core.compute.kubernetes import init_kubernetes

from routes import auth_router, project_router, hub_router, compute_router, tasks_router, skills_router, models_router, proxy_router, conversations_router, metrics_router, agents_router

logger = get_logger()


async def init_app(config: ServerConfig):
    await init_dstack(
        database_url=config.dstack_database_url,
        kubeconfig_path=config.kubeconfig_path,
    )
    init_kubernetes(kubeconfig_path=config.kubeconfig_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from surogate.core.db import init_engine

    config: ServerConfig = getattr(app.state, "config", None)
    engine = init_engine(config.database_url)

    from surogate.core.surogates_client import SurogatesClient
    app.state.surogates = SurogatesClient(config.surogates_database_url)

    await init_app(config)

    factory = get_session_factory()

    from surogate.core.compute.local_tasks import LocalTaskManager
    task_manager = LocalTaskManager(config)
    async with factory() as session:
        await task_manager.reap(session)
    app.state.task_manager = task_manager

    # Start background monitor for dstack runs & local tasks
    from surogate.core.compute.monitor import ServingMonitor
    monitor = ServingMonitor(poll_interval=5.0, task_manager=task_manager)
    monitor.on_transition(notify_transition)
    await monitor.start()
    app.state.serving_monitor = monitor

    # Start background monitor for agent Helm releases
    from surogate.core.compute.agent_monitor import AgentMonitor
    agent_monitor = AgentMonitor(config, poll_interval=5.0)
    agent_monitor.on_transition(notify_transition)
    await agent_monitor.start()
    app.state.agent_monitor = agent_monitor

    yield

    await agent_monitor.stop()
    await monitor.stop()
    await shutdown_dstack()
    await app.state.surogates.close()
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title = "Surogate Server",
    version = "1.0.0",
    description = "Backend API for Surogate",
    lifespan = lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # In production, specify allowed origins
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ============ Register API Routes ============
app.include_router(auth_router, prefix = "/api/auth", tags = ["auth"])
app.include_router(project_router, prefix = "/api/projects", tags = ["projects"])
app.include_router(hub_router, prefix = "/api/hub", tags = ["hub"])
app.include_router(compute_router, prefix = "/api/compute", tags = ["compute"])
app.include_router(compute_router, prefix = "/api/nebius", tags = ["nebius"])
app.include_router(tasks_router, prefix = "/api/tasks", tags = ["tasks"])
app.include_router(skills_router, prefix = "/api/skills", tags = ["skills"])
app.include_router(models_router, prefix = "/api/models", tags = ["models"])
app.include_router(conversations_router, prefix = "/api/conversations", tags = ["conversations"])
app.include_router(metrics_router, prefix = "/api/metrics", tags = ["metrics"])
app.include_router(agents_router, prefix = "/api/agents", tags = ["agents"])

# Mount service proxy with chat-completion observability
from dstack._internal.server.services.proxy.deps import ServerProxyDependencyInjector
app.state.proxy_dependency_injector = ServerProxyDependencyInjector()
app.include_router(proxy_router, prefix = "/proxy/services", tags = ["proxy"])


# ============ WebSocket ============
from fastapi import WebSocket, WebSocketDisconnect
from surogate.server.auth.authentication import verify_ws_token


@app.websocket("/ws/monitor")
async def monitor_ws(ws: WebSocket):
    """Push status-transition events to connected clients (authenticated)."""
    token = ws.query_params.get("token")
    if not token:
        await ws.close(code=4001, reason="Missing token")
        return

    factory = get_session_factory()
    async with factory() as session:
        subject = await verify_ws_token(token, session)
    if subject is None:
        await ws.close(code=4003, reason="Invalid or expired token")
        return

    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)


# ============ Health and System Endpoints ============
@app.post("/api/shutdown")
async def shutdown_server(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Gracefully shut down the Surogate server."""
    import asyncio

    async def _delayed_shutdown():
        await asyncio.sleep(0.2)
        trigger = getattr(request.app.state, "trigger_shutdown", None)
        if trigger is not None:
            trigger()
        else:
            import signal
            import os
            os.kill(os.getpid(), signal.SIGTERM)

    request.app.state._shutdown_task = asyncio.create_task(_delayed_shutdown())
    return {"status": "shutting_down"}



# ============ Serve Frontend ============

def _strip_crossorigin(html_bytes: bytes) -> bytes:
    """Remove ``crossorigin`` attributes from script/link tags."""
    import re as _re

    html = html_bytes.decode("utf-8")
    html = _re.sub(r'\s+crossorigin(?:="[^"]*")?', "", html)
    return html.encode("utf-8")

def setup_frontend(app: FastAPI, build_path: Path):
    """Mount frontend static files (optional)"""
    if not build_path.exists():
        return False

    # Mount assets
    assets_dir = build_path / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory = assets_dir), name = "assets")

    @app.get("/")
    async def serve_root():
        content = (build_path / "index.html").read_bytes()
        content = _strip_crossorigin(content)
        return Response(
            content = content,
            media_type = "text/html",
            headers = {"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api"):
            return {"error": "API endpoint not found"}

        file_path = (build_path / full_path).resolve()

        # Block path traversal — ensure resolved path stays inside build_path
        if not file_path.is_relative_to(build_path.resolve()):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        content = (build_path / "index.html").read_bytes()
        content = _strip_crossorigin(content)
        return Response(
            content = content,
            media_type = "text/html",
            headers = {"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return True
