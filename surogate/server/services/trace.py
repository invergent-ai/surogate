"""Conversation tracker – records every proxied chat-completion round-trip.

Every proxied chat-completion request/response is recorded as a ``ChatTurn``.
Turns sharing the same ``conversation_id`` belong to the same conversation.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.trace import ChatTurn
from surogate.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _canonical(messages: list[dict[str, Any]]) -> str:
    """Deterministic JSON serialisation of a message list."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


_CLIENT_MSG_KEYS = {"role", "content", "tool_calls"}


def _normalize_for_hash(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip assistant messages down to the fields clients send back."""
    out = []
    for msg in messages:
        if msg.get("role") == "assistant":
            out.append({k: v for k, v in msg.items() if k in _CLIENT_MSG_KEYS})
        else:
            out.append(msg)
    return out


def hash_messages(messages: list[dict[str, Any]]) -> str:
    """Hash a full message array (normalized for client round-trip)."""
    return _sha256(_canonical(_normalize_for_hash(messages)))


def _find_last_user_msg(messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the last message with role='user' in the array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def record_turn(
    session: AsyncSession,
    *,
    messages: list[dict[str, Any]],
    assistant_reply: dict[str, Any],
    project_name: str,
    run_name: str,
    model: str,
    is_streaming: bool,
    caller_hash: Optional[str],
    deployed_model_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
    usage: Optional[dict[str, Any]] = None,
    request_body: Optional[dict[str, Any]] = None,
    response_body: Optional[dict[str, Any]] = None,
) -> ChatTurn:
    """Record a completed chat-completion round-trip.

    Returns the newly created ``ChatTurn``.
    """
    # ── Compute hashes ────────────────────────────────────────────────
    parent_hash: Optional[str] = None
    prior_context = messages[:-1] if len(messages) > 1 else []
    if prior_context:
        parent_hash = hash_messages(prior_context)

    full_state = messages + [assistant_reply]
    state_hash = hash_messages(full_state)

    last_user = _find_last_user_msg(messages)
    tail_hash = (
        hash_messages([last_user, assistant_reply])
        if last_user
        else hash_messages([assistant_reply])
    )

    conversation_id = str(uuid.uuid4())

    usage = usage or {}

    # ── Persist ───────────────────────────────────────────────────────
    turn = ChatTurn(
        conversation_id=conversation_id,
        parent_hash=parent_hash,
        state_hash=state_hash,
        tail_hash=tail_hash,
        caller_hash=caller_hash,
        deployed_model_id=deployed_model_id,
        project_name=project_name,
        run_name=run_name,
        model=model,
        is_streaming=is_streaming,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_ms=latency_ms,
        request_body=request_body,
        response_body=response_body,
    )
    session.add(turn)
    await session.commit()

    logger.debug(
        "chat_turn recorded conversation=%s turn=%s",
        conversation_id, turn.id,
    )

    return turn
