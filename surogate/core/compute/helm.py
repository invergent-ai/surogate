"""Helm-based deployment of Surogates agents via pyhelm3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pyhelm3 import Client as HelmClient

from surogate.core.config.server_config import ServerConfig
from surogate.utils.logger import get_logger

logger = get_logger()

# pyhelm3 logs the actual `helm` command it runs at INFO on logger "pyhelm3.command",
# but our surogate logger disables propagation and owns no handler on "pyhelm3".
# Attach our handlers so pyhelm3 output is visible.
_pyhelm_logger = logging.getLogger("pyhelm3")
_pyhelm_logger.setLevel(logging.INFO)
_pyhelm_logger.propagate = False
_surogate_logger = logging.getLogger("surogate")
for _h in _surogate_logger.handlers:
    if _h not in _pyhelm_logger.handlers:
        _pyhelm_logger.addHandler(_h)


def _kubeconfig(config: ServerConfig) -> str:
    return str(Path(config.kubeconfig_path).expanduser())


def _helm_client(config: ServerConfig) -> HelmClient:
    return HelmClient(
        executable=str(Path(config.helm_binary).expanduser()),
        kubeconfig=_kubeconfig(config),
    )


def _release_name(agent_slug: str) -> str:
    return f"agent-{agent_slug}"


def _values_for_agent(
    agent,
    config: ServerConfig,
    llm: dict[str, str],
    org_id: str,
) -> dict[str, Any]:
    """Build the per-agent Helm values overriding the chart defaults."""
    return {
        "agent": {
            "slug": agent.name,
            "domain": config.agent_base_domain,
            # Stamped on every session (``sessions.agent_id``) — the
            # stable UUID, not the slug, so renaming the agent doesn't
            # orphan its sessions and deleting the agent can scrub by
            # agent_id.
            "id": agent.id,
        },
        "org": {"id": org_id},
        "db": {"url": config.agent_surogates_database_url},
        "redis": {"url": config.agent_redis_url},
        "storage": {
            "backend": "s3",
            "endpoint": config.agent_s3_endpoint,
            "accessKey": config.agent_s3_access_key or "",
            "secretKey": config.agent_s3_secret_key or "",
        },
        "llm": llm,
        "sandbox": {"backend": "kubernetes"},
        "api": {"replicas": 1},
        "worker": {"replicas": 1},
        "mcpProxy": {"replicas": 1},
        "autoscaling": {"enabled": False},
        # Tool-specific secrets — the chart renders these as env vars
        # on the worker Deployment only when they are non-empty.
        "tools": {
            "tavilyApiKey": config.tavily_api_key or "",
        },
    }


async def ensure_namespace(namespace: str, config: ServerConfig) -> None:
    """Create the project namespace if missing. Idempotent."""
    from kubernetes import client as k8s_client
    from surogate.core.compute.kubernetes import load_k8s_config

    load_k8s_config(config.kubeconfig_path)
    api = k8s_client.CoreV1Api()
    try:
        api.read_namespace(name=namespace)
    except k8s_client.ApiException as e:
        if e.status != 404:
            raise
        api.create_namespace(
            body=k8s_client.V1Namespace(
                metadata=k8s_client.V1ObjectMeta(name=namespace),
            ),
        )


async def start_agent(
    agent,
    project,
    config: ServerConfig,
    llm: dict[str, str],
    *,
    org_id: str,
) -> None:
    """Helm install/upgrade the Surogates chart into the project namespace.

    ``llm`` is injected into the chart's ``llm`` values (``model``,
    ``baseUrl``, ``apiKey``); the worker's OpenAI client requires a
    non-empty ``apiKey`` even when the target server ignores it.
    ``org_id`` is the surogates tenant the agent belongs to; the
    caller must have ensured the row exists in the surogates DB.
    """
    await ensure_namespace(project.namespace, config)

    client = _helm_client(config)
    chart = await client.get_chart(config.surogates_helm_chart)
    release = _release_name(agent.name)
    values = _values_for_agent(agent, config, llm, org_id)

    logger.info(
        "Deploying agent %s into namespace %s (release=%s)",
        agent.name, project.namespace, release,
    )
    await client.install_or_upgrade_release(
        release,
        chart,
        values,
        namespace=project.namespace,
        atomic=False,
        wait=False,
        timeout="60s",
    )


async def stop_agent(agent, project, config: ServerConfig) -> None:
    """Helm uninstall the agent's release from the project namespace."""
    client = _helm_client(config)
    release = _release_name(agent.name)
    logger.info(
        "Stopping agent %s in namespace %s (release=%s)",
        agent.name, project.namespace, release,
    )
    try:
        await client.uninstall_release(
            release_name=release,
            namespace=project.namespace,
        )
    except Exception:
        logger.warning("Failed to uninstall release %s", release, exc_info=True)
