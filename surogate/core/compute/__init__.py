"""Compute module — dstack initialization and project bridge.

Bootstraps dstack as an embedded library: runs migrations, creates an
admin user, starts the APScheduler background tasks that drive run
state transitions (SUBMITTED → PROVISIONING → RUNNING → DONE/FAILED).

Environment variables are set **before** any dstack import so that
dstack's module-level settings pick them up.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# ── Pre-import env configuration ────────────────────────────────────
# These MUST be set before any dstack._internal import.
# Store dstack data under ~/.surogate/dstack/ instead of ~/.dstack/server/
_surogate_home = str(Path.home() / ".surogate" / "dstack")
os.environ.setdefault("DSTACK_SERVER_DIR", _surogate_home)
os.environ.setdefault("DSTACK_DO_NOT_UPDATE_DEFAULT_PROJECT", "1")
os.environ.setdefault("DSTACK_SERVER_CONFIG_DISABLED", "1")
# TODO: raise back to WARNING once dstack integration is stable
os.environ.setdefault("DSTACK_SERVER_LOG_LEVEL", "DEBUG")
# Prevent dstack's SSH client from offering agent keys — only use the project key file.
# Without this, systems with many SSH keys in the agent exceed MaxAuthTries on the jump pod.
os.environ.pop("SSH_AUTH_SOCK", None)

from surogate.utils.logger import get_logger

logger = get_logger()

# ── Patch dstack create_or_update_repo for asyncpg compat ───────────
# dstack's create_repo uses begin_nested() + session.add() without flush.
# With autoflush=False the INSERT only happens at savepoint commit, but
# the IntegrityError from the commit isn't caught by the try/except.
# We patch create_or_update_repo to check existence first, avoiding the
# broken savepoint path entirely.
def _patch_repos():
    from dstack._internal.server.services import repos as _repos

    _original = _repos.create_or_update_repo

    async def _safe_create_or_update_repo(session, project, repo_id, repo_info):
        # Check if repo already exists — if so, just update
        existing = await _repos.get_repo_model(session, project, repo_id)
        if existing is not None:
            return await _repos.update_repo(session, project, repo_id, repo_info)
        # Doesn't exist — try to create (first run in this project)
        return await _original(session, project, repo_id, repo_info)

    _repos.create_or_update_repo = _safe_create_or_update_repo

_patch_repos()

# ── Module state (set by init_dstack) ───────────────────────────────

_initialized = False
_admin = None              # dstack UserModel
_scheduler = None          # APScheduler instance
_pipeline_manager = None   # Pipeline task manager


async def init_dstack(database_url: str | None = None) -> None:
    """Bootstrap dstack as an embedded library.

    1. Run Alembic migrations for dstack's own DB
    2. Create / retrieve the admin user
    3. Create / retrieve the default project
    4. Register the Kubernetes backend (from current kubeconfig)
    5. Start APScheduler background tasks
    """
    global _initialized, _admin, _scheduler, _pipeline_manager
    if _initialized:
        return

    if database_url:
        os.environ["DSTACK_DATABASE_URL"] = database_url

    from dstack._internal.server.db import Database, get_session_ctx, migrate, override_db
    from dstack._internal.server import settings as dstack_settings
    from dstack._internal.server.services.users import get_or_create_admin_user
    from dstack._internal.server.services.projects import get_or_create_default_project
    from dstack._internal.server.background.scheduled_tasks import start_scheduled_tasks
    # Import triggers EncryptedString.set_encrypt_decrypt() at module level
    import dstack._internal.server.services.encryption  # noqa: F401

    logger.info("Initializing dstack...")

    # dstack's run_sync() calls need a thread pool — match what dstack's
    # own app.py does before any DB work.
    loop = asyncio.get_running_loop()
    if loop._default_executor is None:  # type: ignore[attr-defined]
        loop.set_default_executor(ThreadPoolExecutor(max_workers=128))

    # Resolve the actual DB URL — env var may have been set after settings.py loaded
    db_url = database_url or os.environ.get("DSTACK_DATABASE_URL") or dstack_settings.DATABASE_URL
    logger.info("dstack DB: %s", db_url.split("@")[-1] if "@" in db_url else db_url)

    # dstack creates its DB/engine at import time (before the event loop).
    # Recreate it now with the correct URL and a fresh connection pool.
    override_db(Database(url=db_url))

    # 1. DB migrations
    await migrate()
    # Replace the DB after migration so background tasks get a clean pool.
    override_db(Database(url=db_url))

    # 2. Admin user + default project
    async with get_session_ctx() as session:
        admin, _created = await get_or_create_admin_user(session=session)
        await get_or_create_default_project(session=session, user=admin)

    _admin = admin

    # 3. Background processing
    _scheduler = start_scheduled_tasks()
    logger.info("dstack APScheduler jobs: %s", [j.name for j in _scheduler.get_jobs()])

    from dstack._internal.server.background.scheduled_tasks.probes import PROBES_SCHEDULER
    PROBES_SCHEDULER.start()

    from dstack._internal.settings import FeatureFlags
    if FeatureFlags.PIPELINE_PROCESSING_ENABLED:
        from dstack._internal.server.background.pipeline_tasks import start_pipeline_tasks
        _pipeline_manager = start_pipeline_tasks()
        logger.info("dstack pipeline tasks started")

    _initialized = True
    logger.info("dstack initialised — APScheduler running")


async def ensure_kubernetes_backend(project) -> None:
    """Register the local K8s cluster as a dstack backend on the given project.

    Reads ``~/.kube/config`` (or the in-cluster config) and creates a
    ``kubernetes`` backend so dstack can schedule runs on the cluster.
    Silently skips if the backend already exists or no kubeconfig is found.
    """
    import os
    from pathlib import Path

    from dstack._internal.core.errors import ResourceExistsError
    from dstack._internal.server.db import get_session_ctx
    from dstack._internal.server.services.backends import create_backend

    # Read kubeconfig
    kubeconfig_path = Path(os.environ.get("KUBECONFIG", "~/.kube/config")).expanduser()
    if not kubeconfig_path.exists():
        logger.info("No kubeconfig at %s — skipping K8s backend", kubeconfig_path)
        return

    kubeconfig_data = kubeconfig_path.read_text()

    from dstack._internal.core.backends.kubernetes.models import (
        KubernetesBackendConfigWithCreds,
        KubernetesProxyJumpConfig,
        KubeconfigConfig,
    )

    # Proxy jump lets dstack SSH into pods via an external node address.
    # Set DSTACK_K8S_PROXY_HOST / DSTACK_K8S_PROXY_PORT to enable.
    proxy_jump = None
    proxy_host = os.environ.get("DSTACK_K8S_PROXY_HOST", "localhost")
    if proxy_host:
        proxy_jump = KubernetesProxyJumpConfig(
            hostname=proxy_host,
            port=int(os.environ.get("DSTACK_K8S_PROXY_PORT", "32000")),
        )

    k8s_config = KubernetesBackendConfigWithCreds(
        type="kubernetes",
        kubeconfig=KubeconfigConfig(data=kubeconfig_data),
        namespace=os.environ.get("DSTACK_K8S_NAMESPACE", "dstack"),
        proxy_jump=proxy_jump,
    )

    async with get_session_ctx() as session:
        try:
            await create_backend(session=session, project=project, config=k8s_config)
            logger.info("Registered Kubernetes backend for project '%s'", project.name)
        except ResourceExistsError:
            logger.debug("Kubernetes backend already exists for project '%s'", project.name)
        except Exception as ex:
            logger.warning("Failed to register Kubernetes backend: %s", ex, exc_info=True)

def _invalidate_backends_cache(project) -> None:
    """Evict a project from dstack's in-memory backends cache."""
    from dstack._internal.server.services.backends import _BACKENDS_CACHE
    _BACKENDS_CACHE.pop(project.id, None)


async def ensure_nebius_backend(project, service_account_id: str, public_key_id: str, private_key_content: str) -> None:
    """Register a Nebius backend on the given dstack project.

    Credentials are passed in-memory — nothing is written to disk.
    Silently skips if the backend already exists.
    """
    from dstack._internal.core.errors import ResourceExistsError
    from dstack._internal.server.db import get_session_ctx
    from dstack._internal.server.services.backends import create_backend
    from dstack._internal.core.backends.nebius.models import (
        NebiusBackendConfigWithCreds,
        NebiusServiceAccountCreds,
    )

    creds = NebiusServiceAccountCreds(
        service_account_id=service_account_id,
        public_key_id=public_key_id,
        private_key_content=private_key_content,
    )
    config = NebiusBackendConfigWithCreds(type="nebius", creds=creds)

    _invalidate_backends_cache(project)

    async with get_session_ctx() as session:
        try:
            await create_backend(session=session, project=project, config=config)
            logger.info("Registered Nebius backend for project '%s'", project.name)
        except ResourceExistsError:
            logger.debug("Nebius backend already exists for project '%s'", project.name)
        except Exception as ex:
            logger.warning("Failed to register Nebius backend: %s", ex, exc_info=True)
            raise


async def shutdown_dstack() -> None:
    """Gracefully stop dstack background processing and dispose its DB."""
    global _initialized, _scheduler, _pipeline_manager

    if _pipeline_manager is not None:
        _pipeline_manager.shutdown()
        await _pipeline_manager.drain()
        _pipeline_manager = None

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None

    from dstack._internal.server.db import get_db
    await get_db().engine.dispose()

    _initialized = False
    logger.info("dstack shut down")


# ── Accessors ───────────────────────────────────────────────────────


def get_dstack_admin():
    """Return the dstack admin UserModel (set during init)."""
    if _admin is None:
        raise RuntimeError("dstack not initialised — call init_dstack() first")
    return _admin


async def get_dstack_project(project_name: str):
    """Look up a dstack ProjectModel by name.  Returns None if not found."""
    from dstack._internal.server.db import get_session_ctx
    from dstack._internal.server.services.projects import get_project_model_by_name

    async with get_session_ctx() as session:
        return await get_project_model_by_name(session=session, project_name=project_name)


async def ensure_dstack_project(surogate_project_namespace: str):
    """Get or create a dstack project for a surogate project namespace.

    dstack project names are limited to 50 chars and must match
    ``[a-zA-Z0-9-_]``.  We sanitise the surogate namespace accordingly.

    Returns the dstack ``ProjectModel``.
    """
    import re
    from dstack._internal.server.db import get_session_ctx
    from dstack._internal.server.services.projects import (
        create_project,
        get_project_model_by_name,
        get_project_model_by_name_or_error,
    )

    # Sanitise: keep alphanum / dash / underscore, truncate to 50
    sanitised = re.sub(r"[^a-zA-Z0-9_-]", "-", surogate_project_namespace)[:50]
    if not sanitised:
        sanitised = "default"

    async with get_session_ctx() as session:
        existing = await get_project_model_by_name(session=session, project_name=sanitised)
        if existing is not None:
            return existing

        admin = get_dstack_admin()
        try:
            await create_project(
                session=session,
                user=admin,
                project_name=sanitised,
            )
        except Exception as ex:
            # Race condition or already exists — retrieve it
            logger.warning("Failed to create dstack project: %s", ex, exc_info=True)

        project_model = await get_project_model_by_name_or_error(
            session=session, project_name=sanitised,
        )

    # Register K8s backend on the new project
    await ensure_kubernetes_backend(project_model)
    return project_model
