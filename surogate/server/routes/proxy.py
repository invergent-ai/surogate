"""
Service reverse-proxy with chat-completion observability.

Replaces dstack's built-in service_proxy router so we can intercept,
log, and (later) mangle OpenAI-compatible chat completion requests
flowing through to served models.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Depends, Request, status
from fastapi.datastructures import URL
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from typing_extensions import Annotated

from dstack._internal.proxy.lib.deps import (
    ProxyAuth,
    ProxyAuthContext,
    get_proxy_repo,
    get_service_connection_pool,
)
from dstack._internal.proxy.lib.errors import ProxyError
from dstack._internal.proxy.lib.repo import BaseProxyRepo
from dstack._internal.proxy.lib.services.service_connection import (
    ServiceConnectionPool,
    get_service_replica_client,
)
from dstack._internal.utils.common import concat_url_path

from surogate.core.db.engine import get_session_factory
from surogate.server.services.trace import record_turn
from surogate.utils.logger import get_logger

logger = get_logger()

router = APIRouter()

# ---------------------------------------------------------------------------
# Paths we consider "chat completion" endpoints (OpenAI-compatible)
# ---------------------------------------------------------------------------
_CHAT_COMPLETION_SUFFIXES = (
    "/v1/chat/completions",
    "/chat/completions",
)


def _is_chat_completion(path: str) -> bool:
    lower = path.lower().rstrip("/")
    return any(lower.endswith(s) for s in _CHAT_COMPLETION_SUFFIXES)


# ---------------------------------------------------------------------------
# Redirect bare service URLs to trailing-slash variant
# ---------------------------------------------------------------------------
@router.get("/{project_name}/{run_name}")
@router.post("/{project_name}/{run_name}")
@router.put("/{project_name}/{run_name}")
@router.delete("/{project_name}/{run_name}")
@router.patch("/{project_name}/{run_name}")
@router.head("/{project_name}/{run_name}")
async def redirect_to_service_root(request: Request, project_name: str, run_name: str) -> Response:
    url = URL(str(request.url))
    url = url.replace(path=url.path + "/")
    return RedirectResponse(url, status.HTTP_308_PERMANENT_REDIRECT)


# ---------------------------------------------------------------------------
# Main reverse-proxy handler
# ---------------------------------------------------------------------------
@router.get("/{project_name}/{run_name}/{path:path}")
@router.post("/{project_name}/{run_name}/{path:path}")
@router.put("/{project_name}/{run_name}/{path:path}")
@router.delete("/{project_name}/{run_name}/{path:path}")
@router.patch("/{project_name}/{run_name}/{path:path}")
@router.head("/{project_name}/{run_name}/{path:path}")
@router.options("/{project_name}/{run_name}/{path:path}")
async def service_reverse_proxy(
    project_name: str,
    run_name: str,
    path: str,
    request: Request,
    auth: Annotated[ProxyAuthContext, Depends(ProxyAuth(auto_enforce=False))],
    repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)],
    service_conn_pool: Annotated[ServiceConnectionPool, Depends(get_service_connection_pool)],
) -> Response:
    if "Upgrade" in request.headers:
        raise ProxyError("Upgrading connections is not supported", status.HTTP_400_BAD_REQUEST)

    service = await repo.get_service(project_name, run_name)
    if service is None or not service.replicas:
        raise ProxyError(f"Service {project_name}/{run_name} not found", status.HTTP_404_NOT_FOUND)
    if service.auth:
        await auth.enforce()

    client = await get_service_replica_client(service, repo, service_conn_pool)

    if not service.strip_prefix:
        path = concat_url_path(request.scope.get("root_path", "/"), request.url.path)

    # ── Chat completion observability path ─────────────────────────────
    if _is_chat_completion(path) and request.method == "POST":
        return await _proxy_chat_completion(
            request, path, client, project_name, run_name,
        )

    # ── Generic pass-through ───────────────────────────────────────────
    return await _proxy_passthrough(request, path, client)


# ---------------------------------------------------------------------------
# Generic pass-through (same behaviour as dstack)
# ---------------------------------------------------------------------------

async def _proxy_passthrough(
    request: Request,
    path: str,
    client: httpx.AsyncClient,
) -> Response:
    try:
        upstream_req = await _build_upstream_request(request, path, client)
    except ClientDisconnect:
        raise ProxyError("Client disconnected")

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        raise ProxyError("Timed out requesting upstream", status.HTTP_504_GATEWAY_TIMEOUT)
    except httpx.RequestError:
        raise ProxyError("Error requesting upstream", status.HTTP_502_BAD_GATEWAY)

    return StreamingResponse(
        _stream_response(upstream_resp),
        status_code=upstream_resp.status_code,
        headers=upstream_resp.headers,
    )


# ---------------------------------------------------------------------------
# Chat-completion aware proxy
# ---------------------------------------------------------------------------

async def _proxy_chat_completion(
    request: Request,
    path: str,
    client: httpx.AsyncClient,
    project_name: str,
    run_name: str,
) -> Response:
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    t0 = time.monotonic()

    # Read the request body so we can inspect it
    body = await request.body()

    try:
        req_json = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        req_json = None

    is_streaming = req_json.get("stream", False) if req_json else False
    model_name = req_json.get("model", "") if req_json else ""
    messages = req_json.get("messages", []) if req_json else []

    # Caller identity from auth header
    caller_hash = _hash_caller(request)

    logger.info(
        "chat_completion request_id=%s project=%s service=%s model=%s stream=%s",
        request_id, project_name, run_name, model_name, is_streaming,
    )

    # Build upstream request with the buffered body
    url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
    upstream_req = client.build_request(
        request.method,
        url,
        headers=request.headers,
        content=body,
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        logger.warning(
            "chat_completion timeout request_id=%s project=%s service=%s",
            request_id, project_name, run_name,
        )
        raise ProxyError("Timed out requesting upstream", status.HTTP_504_GATEWAY_TIMEOUT)
    except httpx.RequestError as exc:
        logger.warning(
            "chat_completion upstream_error request_id=%s error=%r", request_id, exc,
        )
        raise ProxyError("Error requesting upstream", status.HTTP_502_BAD_GATEWAY)

    if is_streaming:
        return StreamingResponse(
            _stream_chat_response(
                upstream_resp, request_id, project_name, run_name, model_name, t0,
                messages=messages, caller_hash=caller_hash, req_json=req_json,
            ),
            status_code=upstream_resp.status_code,
            headers=upstream_resp.headers,
        )

    # Non-streaming: read full response, log, then return
    resp_body = await _read_response(upstream_resp)
    elapsed_ms = (time.monotonic() - t0) * 1000

    resp_json = _parse_json(resp_body)
    usage = resp_json.get("usage", {}) if resp_json else {}
    assistant_reply = _extract_assistant_message(resp_json)

    _log_chat_response(request_id, project_name, run_name, model_name, resp_body, elapsed_ms)

    # Record turn (fire-and-forget in background)
    if messages and assistant_reply:
        _schedule_record_turn(
            messages=messages,
            assistant_reply=assistant_reply,
            project_name=project_name,
            run_name=run_name,
            model=model_name,
            is_streaming=False,
            caller_hash=caller_hash,
            latency_ms=elapsed_ms,
            usage=usage,
            request_body=req_json,
            response_body=resp_json,
        )

    return Response(
        content=resp_body,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _stream_chat_response(
    response: httpx.Response,
    request_id: str,
    project_name: str,
    run_name: str,
    model_name: str,
    t0: float,
    *,
    messages: list[dict],
    caller_hash: Optional[str],
    req_json: Optional[dict],
) -> AsyncGenerator[bytes, None]:
    """Yield SSE chunks while accumulating them for logging and tracing."""
    chunks: list[bytes] = []
    try:
        async for chunk in response.aiter_raw():
            chunks.append(chunk)
            yield chunk
    except httpx.RequestError as exc:
        logger.debug("chat_completion stream error request_id=%s: %r", request_id, exc)

    elapsed_ms = (time.monotonic() - t0) * 1000

    try:
        await response.aclose()
    except httpx.RequestError:
        pass

    _log_chat_stream_done(request_id, project_name, run_name, model_name, chunks, elapsed_ms)

    # Reconstruct assistant reply from SSE deltas and record the turn
    usage = _extract_stream_usage(chunks)
    assistant_reply = _assemble_stream_reply(chunks)
    if messages and assistant_reply:
        _schedule_record_turn(
            messages=messages,
            assistant_reply=assistant_reply,
            project_name=project_name,
            run_name=run_name,
            model=model_name,
            is_streaming=True,
            caller_hash=caller_hash,
            latency_ms=elapsed_ms,
            usage=usage,
            request_body=req_json,
            response_body=None,
        )


async def _stream_response(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Plain streaming pass-through (non-chat)."""
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    except httpx.RequestError:
        pass
    try:
        await response.aclose()
    except httpx.RequestError:
        pass


async def _read_response(response: httpx.Response) -> bytes:
    """Read full response body and close."""
    body = b""
    try:
        async for chunk in response.aiter_raw():
            body += chunk
    finally:
        try:
            await response.aclose()
        except httpx.RequestError:
            pass
    return body


# ---------------------------------------------------------------------------
# Upstream request building
# ---------------------------------------------------------------------------

async def _build_upstream_request(
    downstream_request: Request,
    path: str,
    client: httpx.AsyncClient,
) -> httpx.Request:
    url = httpx.URL(path=path, query=downstream_request.url.query.encode("utf-8"))
    stream = await _FastAPIBodyAdaptor(downstream_request.stream()).get_stream()
    client.cookies.clear()
    return client.build_request(
        downstream_request.method, url, headers=downstream_request.headers, content=stream,
    )


class _FastAPIBodyAdaptor:
    """Convert empty FastAPI body streams to None (same as dstack)."""

    def __init__(self, stream: AsyncIterator[bytes]) -> None:
        self._stream = stream

    async def get_stream(self) -> Optional[AsyncGenerator[bytes, None]]:
        try:
            first = await self._stream.__anext__()
        except (StopAsyncIteration, ClientDisconnect):
            return None
        if first == b"":
            return None
        return self._rest(first)

    async def _rest(self, first: bytes) -> AsyncGenerator[bytes, None]:
        yield first
        try:
            async for chunk in self._stream:
                yield chunk
        except ClientDisconnect:
            pass


# ---------------------------------------------------------------------------
# Observability logging
# ---------------------------------------------------------------------------

def _log_chat_response(
    request_id: str,
    project_name: str,
    run_name: str,
    model_name: str,
    resp_body: bytes,
    elapsed_ms: float,
) -> None:
    usage = _extract_usage(resp_body)
    logger.info(
        "chat_completion done request_id=%s project=%s service=%s model=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s latency_ms=%.1f",
        request_id, project_name, run_name, model_name,
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
        usage.get("total_tokens", "?"),
        elapsed_ms,
    )


def _log_chat_stream_done(
    request_id: str,
    project_name: str,
    run_name: str,
    model_name: str,
    chunks: list[bytes],
    elapsed_ms: float,
) -> None:
    # Try to extract usage from the final SSE chunk (vLLM / OpenAI include it)
    usage = _extract_stream_usage(chunks)
    logger.info(
        "chat_completion stream_done request_id=%s project=%s service=%s model=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s latency_ms=%.1f",
        request_id, project_name, run_name, model_name,
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
        usage.get("total_tokens", "?"),
        elapsed_ms,
    )


def _extract_usage(resp_body: bytes) -> dict:
    """Extract usage from a non-streaming chat completion response."""
    try:
        data = json.loads(resp_body)
        return data.get("usage", {})
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _extract_stream_usage(chunks: list[bytes]) -> dict:
    """Extract usage from SSE stream (typically in the last data chunk)."""
    raw = b"".join(chunks).decode("utf-8", errors="replace")
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
            usage = obj.get("usage")
            if usage:
                return usage
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


# ---------------------------------------------------------------------------
# Caller identity
# ---------------------------------------------------------------------------

def _hash_caller(request: Request) -> Optional[str]:
    """Derive a stable caller fingerprint from the Authorization header."""
    auth = request.headers.get("authorization", "")
    if not auth:
        return None
    return hashlib.sha256(auth.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def _parse_json(body: bytes) -> Optional[dict]:
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _extract_assistant_message(resp_json: Optional[dict]) -> Optional[dict[str, Any]]:
    """Extract the assistant message from a non-streaming chat completion response."""
    if not resp_json:
        return None
    choices = resp_json.get("choices", [])
    if not choices:
        return None
    return choices[0].get("message")


def _assemble_stream_reply(chunks: list[bytes]) -> Optional[dict[str, Any]]:
    """Reassemble a complete assistant message from SSE stream deltas."""
    raw = b"".join(chunks).decode("utf-8", errors="replace")
    role = "assistant"
    content_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            if "role" in delta:
                role = delta["role"]
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls_by_idx:
                    tool_calls_by_idx[idx] = {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_by_idx[idx]
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    entry["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]

    if not content_parts and not tool_calls_by_idx:
        return None

    msg: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
    if tool_calls_by_idx:
        msg["tool_calls"] = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
    return msg


# ---------------------------------------------------------------------------
# Background turn recording
# ---------------------------------------------------------------------------

def _schedule_record_turn(**kwargs: Any) -> None:
    """Fire-and-forget: record a ChatTurn in the background."""
    asyncio.create_task(_do_record_turn(**kwargs))


async def _do_record_turn(**kwargs: Any) -> None:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await record_turn(session, **kwargs)
    except Exception:
        logger.warning("failed to record chat turn", exc_info=True)
