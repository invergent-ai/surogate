"""Background monitor for agent Helm releases.

Polls Kubernetes every ``poll_interval`` seconds for the Deployments
owned by each agent's Helm release (``{release}-api``, ``-worker``,
``-mcp-proxy``), derives an aggregate status, and — when it changes —
updates the ``agents`` row and fires transition callbacks (which
broadcast to connected WebSocket clients via
``server.notifier.manager``).

Mirrors :class:`surogate.core.compute.monitor.ServingMonitor` in shape
so the UI can consume both streams through the same channel.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from kubernetes import client as k8s_client
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from surogate.core.compute.kubernetes import load_k8s_config
from surogate.core.config.server_config import ServerConfig
from surogate.core.db.engine import get_session_factory
from surogate.core.db.models.operate import Agent, AgentStatus
from surogate.core.db.repository import agents as agent_repo
from surogate.utils.logger import get_logger

logger = get_logger()


TransitionCallback = Callable[
    [str, str, str, str, str, dict[str, Any]], Awaitable[None]
]

_COMPONENTS = ("api", "worker", "mcp-proxy")
# Statuses the monitor actively reconciles.  ``stopped`` agents are
# left alone — the user must explicitly restart them.
_ACTIVE_STATUSES = {AgentStatus.deploying, AgentStatus.running, AgentStatus.error}


class AgentMonitor:
    """Periodically reconciles agent status from Kubernetes.

    Usage::

        monitor = AgentMonitor(config, poll_interval=5.0)
        monitor.on_transition(notify_transition)
        await monitor.start()
        ...
        await monitor.stop()
    """

    def __init__(
        self,
        config: ServerConfig,
        poll_interval: float = 5.0,
    ) -> None:
        self._config = config
        self._poll_interval = poll_interval
        self._task: Optional[asyncio.Task] = None
        self._callbacks: list[TransitionCallback] = []
        load_k8s_config(config.kubeconfig_path)
        self._apps_api = k8s_client.AppsV1Api()

    # ── Public API ──────────────────────────────────────────────────

    def on_transition(self, cb: TransitionCallback) -> None:
        self._callbacks.append(cb)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "AgentMonitor started (interval=%ss)", self._poll_interval,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("AgentMonitor stopped")

    # ── Internal ────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("AgentMonitor tick failed", exc_info=True)
            await asyncio.sleep(self._poll_interval)

    async def _tick(self) -> None:
        factory = get_session_factory()
        async with factory() as session:
            stmt = (
                select(Agent)
                .where(Agent.status.in_(_ACTIVE_STATUSES))
                .options(
                    selectinload(Agent.project),
                    selectinload(Agent.model),
                    selectinload(Agent.created_by),
                )
            )
            agents = (await session.execute(stmt)).scalars().all()

            for agent in agents:
                if agent.project is None:
                    continue
                new_status = await self._derive_status(
                    namespace=agent.project.namespace,
                    agent_slug=agent.name,
                )
                if new_status is None or new_status == agent.status:
                    continue

                old_status = agent.status
                await agent_repo.update_agent(
                    session, agent.id, status=new_status,
                )
                agent.status = new_status
                data = agent_view(agent, self._config.agent_base_domain)
                await self._fire_callbacks(
                    "agent", agent.id, agent.name,
                    old_status.value, new_status.value, data,
                )

    async def _derive_status(
        self, namespace: str, agent_slug: str,
    ) -> Optional[AgentStatus]:
        """Return the aggregate status of an agent's Deployments.

        * all Deployments Available → ``running``
        * any Deployment reports ``ReplicaFailure`` or
          ``Progressing=False`` → ``error``
        * otherwise → ``deploying``
        * namespace missing or no Deployments found yet → ``None``
          (leave the DB alone — the install may still be propagating)
        """
        release = f"agent-{agent_slug}"
        expected = [f"{release}-{c}" for c in _COMPONENTS]

        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._apps_api.read_namespaced_deployment_status,
                    name, namespace,
                )
                for name in expected
            ),
            return_exceptions=True,
        )

        found = 0
        all_available = True
        any_failed = False

        for name, dep in zip(expected, results):
            if isinstance(dep, k8s_client.ApiException):
                if dep.status == 404:
                    all_available = False
                    continue
                logger.debug(
                    "AgentMonitor: k8s read failed for %s/%s",
                    namespace, name, exc_info=dep,
                )
                return None
            if isinstance(dep, BaseException):
                logger.debug(
                    "AgentMonitor: k8s read failed for %s/%s",
                    namespace, name, exc_info=dep,
                )
                return None
            found += 1

            desired = (dep.spec.replicas or 0) if dep.spec else 0
            available = (
                (dep.status.available_replicas or 0) if dep.status else 0
            )
            if desired == 0 or available < desired:
                all_available = False

            for cond in (dep.status.conditions or []) if dep.status else []:
                if cond.type == "ReplicaFailure" and cond.status == "True":
                    any_failed = True
                if (
                    cond.type == "Progressing"
                    and cond.status == "False"
                    and cond.reason == "ProgressDeadlineExceeded"
                ):
                    any_failed = True

        if found == 0:
            return None
        if any_failed:
            return AgentStatus.error
        if all_available and found == len(expected):
            return AgentStatus.running
        return AgentStatus.deploying

    async def _fire_callbacks(
        self, entity_type: str, entity_id: str, name: str,
        old_status: str, new_status: str, data: dict[str, Any],
    ) -> None:
        for cb in self._callbacks:
            try:
                await cb(
                    entity_type, entity_id, name,
                    old_status, new_status, data,
                )
            except Exception:
                logger.warning(
                    "Transition callback failed for %s %s",
                    entity_type, name, exc_info=True,
                )


def agent_view(agent: Agent, agent_base_domain: str = "") -> dict[str, Any]:
    """Build an ``AgentResponse``-shaped dict from an Agent ORM object."""
    endpoint = agent.endpoint or ""
    if not endpoint and agent_base_domain:
        endpoint = f"https://{agent.name}.{agent_base_domain}"
    return {
        "id": agent.id,
        "project_id": agent.project_id,
        "project_name": agent.project.name if agent.project else "",
        "name": agent.name,
        "harness": agent.harness,
        "display_name": agent.display_name,
        "description": agent.description or "",
        "version": agent.version,
        "model_id": agent.model_id,
        "model_name": agent.model.name if agent.model else "",
        "status": agent.status.value,
        "replicas": agent.replicas,
        "image": agent.image or "",
        "endpoint": endpoint,
        "system_prompt": agent.system_prompt or "",
        "env_vars": agent.env_vars,
        "resources": agent.resources,
        "created_by_id": agent.created_by_id,
        "created_by_username": (
            agent.created_by.username if agent.created_by else ""
        ),
        "hub_ref": agent.hub_ref,
        "created_at": str(agent.created_at) if agent.created_at else None,
    }
