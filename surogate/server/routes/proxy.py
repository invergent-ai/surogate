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
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.datastructures import URL
from sqlalchemy.ext.asyncio import AsyncSession
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
from surogate.server.services.metrics import record_metric
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
# Proxy-model upstream routing (OpenRouter / OpenAI-compatible URL)
#
# IMPORTANT: these routes MUST be registered before the dstack catch-all
# /{project_name}/{run_name}/{path:path} routes below, otherwise FastAPI
# would match "_model" as a project name.
# ---------------------------------------------------------------------------

_UPSTREAM_BASES: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api",
}

from surogate.core.db.engine import get_session as _get_session
from surogate.core.db.repository import compute as _compute_repo
from surogate.server.auth.authentication import (
    require_model_access as _require_model_access,
    security as _bearer_security,
)
from fastapi.security import HTTPAuthorizationCredentials as _HTTPAuthCreds


class _ProxyModelInfo:
    __slots__ = ("client", "headers", "base_url", "model_id")

    def __init__(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        base_url: str,
        model_id: str,
    ):
        self.client = client
        self.headers = headers
        self.base_url = base_url
        self.model_id = model_id


async def _get_proxy_model_info(
    model_id: str,
    session: AsyncSession,
) -> _ProxyModelInfo:
    """Build an httpx client + headers targeting the upstream API.

    Resolves the upstream for every model type:

    * ``engine == "openrouter"`` — upstream is OpenRouter; the platform-
      held key is injected as a Bearer header.
    * ``engine == "openai_compat"`` — upstream is a user-provided
      OpenAI-compatible URL with a user-provided key.
    * Otherwise (dstack-served models with ``serving_service_id``) —
      upstream is the in-cluster service endpoint; no auth header.

    Raises HTTPException if the model has no resolvable upstream.
    """
    from surogate.core.db.models.compute import ServingService

    model = await _compute_repo.get_deployed_model(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    engine = model.engine
    serving_config = model.serving_config or {}
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if engine == "openrouter":
        api_key = serving_config.get("api_key", "")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="No API key configured for this model",
            )
        base_url = _UPSTREAM_BASES["openrouter"]
        headers["Authorization"] = f"Bearer {api_key}"
    elif engine == "openai_compat":
        api_key = serving_config.get("api_key", "")
        base_url = serving_config.get("endpoint", "")
        if not base_url:
            raise HTTPException(
                status_code=400,
                detail="No endpoint configured for this model",
            )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    elif model.serving_service_id:
        svc = await session.get(ServingService, model.serving_service_id)
        if svc is None or not svc.endpoint:
            raise HTTPException(
                status_code=400,
                detail="Model has no serving endpoint yet",
            )
        base_url = svc.endpoint.rstrip("/")
    else:
        raise HTTPException(
            status_code=400,
            detail="Model has no usable upstream (no engine, no serving service)",
        )

    # Fresh client per request keeps the implementation simple;
    # revisit pooling if proxy QPS becomes a bottleneck.
    client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout=120.0, connect=10.0),
    )

    return _ProxyModelInfo(client, headers, base_url, model_id)


@router.get("/_model/{model_id}/{path:path}")
@router.post("/_model/{model_id}/{path:path}")
@router.put("/_model/{model_id}/{path:path}")
@router.delete("/_model/{model_id}/{path:path}")
@router.patch("/_model/{model_id}/{path:path}")
@router.head("/_model/{model_id}/{path:path}")
@router.options("/_model/{model_id}/{path:path}")
async def proxy_model_upstream(
    model_id: str,
    path: str,
    request: Request,
    credentials: _HTTPAuthCreds = Depends(_bearer_security),
    session: AsyncSession = Depends(_get_session),
) -> Response:
    await _require_model_access(model_id, credentials, session)
    """Proxy requests for upstream models (OpenRouter, OpenAI-compat URL).

    Streams the upstream response directly to the client, with
    chat-completion observability (metrics + tracing).
    """
    info = await _get_proxy_model_info(model_id, session)
    upstream_url = f"{info.base_url.rstrip('/')}/{path.lstrip('/')}"

    body = await request.body()
    req_json = json.loads(body) if body else {}
    is_streaming = req_json.get("stream", False)
    model_name = req_json.get("model", "")
    messages = req_json.get("messages", [])
    caller_hash = _hash_caller(request)
    t0 = time.monotonic()

    upstream_resp = await info.client.send(
        info.client.build_request(
            request.method,
            upstream_url,
            headers=info.headers,
            content=body,
        ),
        stream=True,
    )

    if is_streaming:
        async def generate():
            chunks: list[bytes] = []
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    chunks.append(chunk)
                    yield chunk
            finally:
                await upstream_resp.aclose()
                await info.client.aclose()

                # Record observability in the background
                elapsed_ms = (time.monotonic() - t0) * 1000
                assistant_reply = _assemble_stream_reply(chunks)
                prompt_tokens = _estimate_prompt_tokens(messages)
                completion_tokens = _count_stream_completion_tokens(chunks)
                _schedule_record_metric(
                    project_name="_proxy", run_name=model_id,
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=elapsed_ms,
                    status_code=upstream_resp.status_code,
                    is_streaming=True,
                )
                if messages and assistant_reply:
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    synthetic_resp = {
                        "choices": [{"index": 0, "message": assistant_reply, "finish_reason": "stop"}],
                        "usage": usage,
                    }
                    _schedule_record_turn(
                        messages=messages, assistant_reply=assistant_reply,
                        project_name="_proxy", run_name=model_id,
                        model=model_name, is_streaming=True,
                        deployed_model_id=model_id,
                        caller_hash=caller_hash, latency_ms=elapsed_ms,
                        usage=usage,
                        request_body=req_json, response_body=synthetic_resp,
                    )

        return StreamingResponse(
            generate(),
            status_code=upstream_resp.status_code,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    try:
        resp_body = b""
        async for chunk in upstream_resp.aiter_bytes():
            resp_body += chunk
    finally:
        await upstream_resp.aclose()
        await info.client.aclose()

    elapsed_ms = (time.monotonic() - t0) * 1000
    resp_json = _parse_json(resp_body)
    assistant_reply = _extract_assistant_message(resp_json)
    prompt_tokens = _estimate_prompt_tokens(messages)
    completion_tokens = _count_response_completion_tokens(resp_json)
    _schedule_record_metric(
        project_name="_proxy", run_name=model_id, model=model_name,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=elapsed_ms, status_code=upstream_resp.status_code,
        is_streaming=False,
    )
    if messages and assistant_reply:
        _schedule_record_turn(
            messages=messages, assistant_reply=assistant_reply,
            project_name="_proxy", run_name=model_id,
            model=model_name, is_streaming=False,
            deployed_model_id=model_id,
            caller_hash=caller_hash, latency_ms=elapsed_ms,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            request_body=req_json, response_body=resp_json,
        )

    return Response(
        content=resp_body,
        status_code=upstream_resp.status_code,
        media_type="application/json",
    )


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
        # Resolve deployed model ID from dstack run_name (best-effort)
        deployed_model_id: Optional[str] = None
        try:
            factory = get_session_factory()
            async with factory() as obs_session:
                dm = await _compute_repo.get_deployed_model_by_run_name(obs_session, run_name)
                if dm is not None:
                    deployed_model_id = dm.id
        except Exception:
            pass
        return await _proxy_chat_completion(
            request, path, client, project_name, run_name,
            deployed_model_id=deployed_model_id,
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
    *,
    deployed_model_id: Optional[str] = None,
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
        _schedule_record_metric(
            project_name=project_name, run_name=run_name, model=model_name,
            latency_ms=(time.monotonic() - t0) * 1000,
            status_code=504, is_streaming=is_streaming,
        )
        raise ProxyError("Timed out requesting upstream", status.HTTP_504_GATEWAY_TIMEOUT)
    except httpx.RequestError as exc:
        logger.warning(
            "chat_completion upstream_error request_id=%s error=%r", request_id, exc,
        )
        _schedule_record_metric(
            project_name=project_name, run_name=run_name, model=model_name,
            latency_ms=(time.monotonic() - t0) * 1000,
            status_code=502, is_streaming=is_streaming,
        )
        raise ProxyError("Error requesting upstream", status.HTTP_502_BAD_GATEWAY)

    if is_streaming:
        return StreamingResponse(
            _stream_chat_response(
                upstream_resp, request_id, project_name, run_name, model_name, t0,
                messages=messages, caller_hash=caller_hash, req_json=req_json,
                deployed_model_id=deployed_model_id,
            ),
            status_code=upstream_resp.status_code,
            headers=upstream_resp.headers,
        )

    # Non-streaming: read full response, log, then return
    resp_body = await _read_response(upstream_resp)
    elapsed_ms = (time.monotonic() - t0) * 1000

    resp_json = _parse_json(resp_body)
    assistant_reply = _extract_assistant_message(resp_json)

    prompt_tokens = _estimate_prompt_tokens(messages)
    completion_tokens = _count_response_completion_tokens(resp_json)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    # Record metric (fire-and-forget)
    _schedule_record_metric(
        project_name=project_name,
        run_name=run_name,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=elapsed_ms,
        status_code=upstream_resp.status_code,
        is_streaming=False,
    )

    # Record turn (fire-and-forget in background)
    if messages and assistant_reply:
        _schedule_record_turn(
            messages=messages,
            assistant_reply=assistant_reply,
            project_name=project_name,
            run_name=run_name,
            model=model_name,
            is_streaming=False,
            deployed_model_id=deployed_model_id,
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
    deployed_model_id: Optional[str] = None,
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

    # Reconstruct assistant reply from SSE deltas and record the turn
    assistant_reply = _assemble_stream_reply(chunks)

    prompt_tokens = _estimate_prompt_tokens(messages)
    completion_tokens = _count_stream_completion_tokens(chunks)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    # Record metric (fire-and-forget)
    _schedule_record_metric(
        project_name=project_name,
        run_name=run_name,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=elapsed_ms,
        status_code=response.status_code,
        is_streaming=True,
    )

    if messages and assistant_reply:
        # Build a synthetic response body so the detail view can show messages
        synthetic_resp = {
            "choices": [{"index": 0, "message": assistant_reply, "finish_reason": "stop"}],
            "usage": usage,
        }
        _schedule_record_turn(
            messages=messages,
            assistant_reply=assistant_reply,
            project_name=project_name,
            run_name=run_name,
            model=model_name,
            is_streaming=True,
            deployed_model_id=deployed_model_id,
            caller_hash=caller_hash,
            latency_ms=elapsed_ms,
            usage=usage,
            request_body=req_json,
            response_body=synthetic_resp,
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
# Token counting (backend-independent)
# ---------------------------------------------------------------------------

def _estimate_prompt_tokens(messages: list[dict]) -> int:
    """Estimate prompt token count from the chat messages.

    Uses a ~4 chars/token heuristic which is reasonable across most
    LLM tokenizers (BPE averages 3.5–4.5 chars/token for English).
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Multimodal messages: list of content parts
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", ""))
        # Count role/name overhead (~4 tokens per message for framing)
        total_chars += 16
    return max(1, total_chars // 4)


def _count_stream_completion_tokens(chunks: list[bytes]) -> int:
    """Count completion tokens by counting SSE content deltas.

    Each SSE delta with a non-empty ``content`` or ``reasoning_content``
    field corresponds to one token emitted by the model.
    """
    raw = b"".join(chunks).decode("utf-8", errors="replace")
    count = 0
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
            if delta.get("content"):
                count += 1
            if delta.get("reasoning_content"):
                count += 1
    return max(1, count) if count > 0 else 0


def _count_response_completion_tokens(resp_json: Optional[dict]) -> int:
    """Count completion tokens from a non-streaming response body."""
    if not resp_json:
        return 0
    choices = resp_json.get("choices", [])
    if not choices:
        return 0
    msg = choices[0].get("message", {})
    total_chars = 0
    content = msg.get("content", "")
    if content:
        total_chars += len(content)
    reasoning = msg.get("reasoning_content", "")
    if reasoning:
        total_chars += len(reasoning)
    return max(1, total_chars // 4) if total_chars > 0 else 0


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
    """Reassemble a complete assistant message from SSE stream deltas.

    Handles both standard ``content`` and ``reasoning_content`` fields
    (used by Qwen3, DeepSeek and other thinking models).
    """
    raw = b"".join(chunks).decode("utf-8", errors="replace")
    role = "assistant"
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
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
            if "reasoning_content" in delta and delta["reasoning_content"]:
                reasoning_parts.append(delta["reasoning_content"])
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

    if not content_parts and not reasoning_parts and not tool_calls_by_idx:
        return None

    msg: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
    if reasoning_parts:
        msg["reasoning_content"] = "".join(reasoning_parts)
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


def _schedule_record_metric(**kwargs: Any) -> None:
    """Fire-and-forget: record a ModelMetric in the background."""
    asyncio.create_task(_do_record_metric(**kwargs))


async def _do_record_metric(**kwargs: Any) -> None:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await record_metric(session, **kwargs)
    except Exception:
        logger.warning("failed to record model metric", exc_info=True)


