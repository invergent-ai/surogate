"""Reverse-proxy route for SkyPilot serving services.

Instead of SkyPilot spawning a separate load-balancer process per service,
we proxy requests through the Surogate server itself.  This gives us a
single entry point for auth, observability, and custom routing.

Architecture
------------
Each SkyPilot serving service has a *controller* process (FastAPI on a local
port) that knows which replicas are ready.  We periodically sync with every
active controller (``/controller/load_balancer_sync``) to maintain an
in-process map of ``service_name -> [replica_url, ...]`` and forward
incoming requests to a replica chosen by a least-load policy.

Route:  ``/api/models/serve/{service_name}/{path:path}``
"""

import asyncio
import time
import threading
from collections import defaultdict
from typing import Any, Optional

import aiohttp
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from surogate.utils.logger import get_logger

logger = get_logger()

router = APIRouter()


# ---------------------------------------------------------------------------
# Registry: service_name -> proxy state
# ---------------------------------------------------------------------------

class _ReplicaState:
    """Per-service state kept in-process."""

    __slots__ = ("controller_url", "replicas", "load", "lock", "client_pool")

    def __init__(self, controller_url: str) -> None:
        self.controller_url = controller_url
        self.replicas: list[str] = []
        self.load: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.client_pool: dict[str, httpx.AsyncClient] = {}


# service_name -> _ReplicaState
_services: dict[str, _ReplicaState] = {}
_sync_task: Optional[asyncio.Task] = None

_SYNC_INTERVAL = 20  # seconds – matches SkyPilot's LB_CONTROLLER_SYNC_INTERVAL


def register_service(service_name: str, controller_port: int) -> None:
    """Register a service for proxying (called after controller starts)."""
    controller_url = f"http://127.0.0.1:{controller_port}"
    _services[service_name] = _ReplicaState(controller_url)
    logger.info("Proxy: registered service %s -> %s", service_name, controller_url)


def unregister_service(service_name: str) -> None:
    """Remove a service from the proxy registry."""
    state = _services.pop(service_name, None)
    if state is not None:
        # Close clients in the background
        for client in state.client_pool.values():
            asyncio.create_task(client.aclose())
        logger.info("Proxy: unregistered service %s", service_name)


# ---------------------------------------------------------------------------
# Background sync with controllers
# ---------------------------------------------------------------------------

async def start_sync_loop() -> None:
    global _sync_task
    if _sync_task is not None:
        return
    _sync_task = asyncio.create_task(_sync_loop())
    logger.info("Proxy sync loop started (interval=%ss)", _SYNC_INTERVAL)


async def stop_sync_loop() -> None:
    global _sync_task
    if _sync_task is None:
        return
    _sync_task.cancel()
    try:
        await _sync_task
    except asyncio.CancelledError:
        pass
    _sync_task = None
    # Close all httpx clients
    for state in _services.values():
        for client in state.client_pool.values():
            await client.aclose()
        state.client_pool.clear()
    logger.info("Proxy sync loop stopped")


async def _sync_loop() -> None:
    await asyncio.sleep(5)  # initial grace period for controllers to boot
    while True:
        try:
            await _sync_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Proxy sync tick failed", exc_info=True)
        await asyncio.sleep(_SYNC_INTERVAL)


async def _sync_all() -> None:
    """Sync replica lists from every registered controller."""
    tasks = [
        _sync_one(name, state) for name, state in list(_services.items())
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _sync_one(service_name: str, state: _ReplicaState) -> None:
    async with aiohttp.ClientSession() as session:
        try:
            payload: dict[str, Any] = {
                "request_aggregator": {"timestamps": []},
            }
            async with session.post(
                f"{state.controller_url}/controller/load_balancer_sync",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                replica_info = data.get("replica_info", {})
                new_urls = list(replica_info.keys())
        except Exception as exc:
            logger.debug(
                "Proxy: failed to sync %s: %s", service_name, exc
            )
            return

    with state.lock:
        old_urls = set(state.replicas)
        state.replicas = new_urls

        # Create clients for new replicas
        for url in new_urls:
            if url not in state.client_pool:
                state.client_pool[url] = httpx.AsyncClient(base_url=url)

        # Close clients for removed replicas
        removed = old_urls - set(new_urls)
        clients_to_close = [state.client_pool.pop(url) for url in removed if url in state.client_pool]
        for url in removed:
            state.load.pop(url, None)

    for client in clients_to_close:
        await client.aclose()

    if new_urls:
        logger.debug("Proxy: %s has %d replicas", service_name, len(new_urls))


# ---------------------------------------------------------------------------
# Replica selection (least-load)
# ---------------------------------------------------------------------------

def _select_replica(state: _ReplicaState) -> Optional[str]:
    with state.lock:
        if not state.replicas:
            return None
        return min(state.replicas, key=lambda r: state.load[r])


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.api_route(
    "/{service_name}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_to_service(service_name: str, path: str, request: Request):
    state = _services.get(service_name)
    if state is None:
        raise HTTPException(404, f"Service '{service_name}' not registered for proxying")

    replica_url = _select_replica(state)
    if replica_url is None:
        raise HTTPException(503, f"No ready replicas for service '{service_name}'")

    with state.lock:
        client = state.client_pool.get(replica_url)
    if client is None:
        raise HTTPException(503, f"No connection to replica {replica_url}")

    # Increment load
    with state.lock:
        state.load[replica_url] += 1

    try:
        target = httpx.URL(path=f"/{path}", query=request.url.query.encode("utf-8"))
        body = await request.body()

        proxy_req = client.build_request(
            request.method,
            target,
            headers=request.headers.raw,
            content=body,
            timeout=120,
        )
        proxy_resp = await client.send(proxy_req, stream=True)

        async def _decrement_and_close():
            await proxy_resp.aclose()
            with state.lock:
                state.load[replica_url] -= 1

        return StreamingResponse(
            content=proxy_resp.aiter_raw(),
            status_code=proxy_resp.status_code,
            headers=dict(proxy_resp.headers),
            background=_decrement_and_close,
        )
    except Exception:
        # Decrement on error
        with state.lock:
            state.load[replica_url] -= 1
        raise HTTPException(502, f"Failed to proxy to replica {replica_url}")
