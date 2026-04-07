"""Conversation tracker – hash-linked-list over chat-completion turns.

Every proxied chat-completion request/response is recorded as a ``ChatTurn``.
Turns are linked into conversation DAGs via dual-key indexing:

**Pass 1 (strict):** ``parent_hash = sha256(messages[:-1])`` must match a
prior turn's ``state_hash``.  This is the normal, uncompacted path.

**Pass 2 (tail anchor):** When Pass 1 fails (agent compacted history), we
hash the last ``[user, assistant]`` pair from the incoming context and match
it against stored ``tail_hash`` values.  After a match the chain is *healed*
so that future lookups go through Pass 1 again.

Branching (regenerations, A/B tests) is natural: same parent, different child.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Optional

from sqlalchemy import select
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


def hash_messages(messages: list[dict[str, Any]]) -> str:
    """Hash a full message array."""
    return _sha256(_canonical(messages))


def _compute_tail_hash(
    last_user_msg: dict[str, Any],
    assistant_reply: dict[str, Any],
) -> str:
    """Hash the [user, assistant] pair that anchors the conversation tail."""
    return _sha256(_canonical([last_user_msg, assistant_reply]))


def _find_last_user_msg(messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the last message with role='user' in the array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


# ---------------------------------------------------------------------------
# Parent lookup
# ---------------------------------------------------------------------------

async def _pass1_strict(
    session: AsyncSession,
    parent_hash: str,
) -> Optional[ChatTurn]:
    """Pass 1: find a turn whose state_hash matches our parent_hash."""
    stmt = select(ChatTurn).where(ChatTurn.state_hash == parent_hash).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _pass2_tail_anchor(
    session: AsyncSession,
    messages: list[dict[str, Any]],
    caller_hash: Optional[str],
) -> Optional[ChatTurn]:
    """Pass 2: compaction recovery via tail anchor.

    Extract the last user+assistant pair from ``messages`` (excluding the
    newest user message which is the current prompt).  Hash the pair and
    look it up in the ``tail_hash`` index.
    """
    # messages is everything *before* the current user prompt (messages[:-1]
    # was already stripped by the caller).  We need the last assistant msg
    # and the user msg that preceded it.
    last_assistant: Optional[dict] = None
    preceding_user: Optional[dict] = None

    for msg in reversed(messages):
        if last_assistant is None and msg.get("role") == "assistant":
            last_assistant = msg
        elif last_assistant is not None and msg.get("role") == "user":
            preceding_user = msg
            break

    if last_assistant is None or preceding_user is None:
        return None

    anchor = _compute_tail_hash(preceding_user, last_assistant)

    stmt = select(ChatTurn).where(ChatTurn.tail_hash == anchor)
    if caller_hash:
        stmt = stmt.where(ChatTurn.caller_hash == caller_hash)
    stmt = stmt.order_by(ChatTurn.created_at.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _heal_chain(
    session: AsyncSession,
    parent_turn: ChatTurn,
    new_state_hash: str,
) -> None:
    """After a compaction match, update the parent's state_hash to the
    compacted context so that future Pass-1 lookups succeed directly."""
    parent_turn.state_hash = new_state_hash
    await session.flush()


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
    latency_ms: Optional[float] = None,
    usage: Optional[dict[str, Any]] = None,
    request_body: Optional[dict[str, Any]] = None,
    response_body: Optional[dict[str, Any]] = None,
) -> ChatTurn:
    """Record a completed chat-completion round-trip and link it into the
    conversation DAG.

    Returns the newly created ``ChatTurn``.
    """
    # ── Compute hashes ────────────────────────────────────────────────

    # parent_hash: everything the client had before the newest user message
    parent_hash: Optional[str] = None
    prior_context = messages[:-1] if len(messages) > 1 else []
    if prior_context:
        parent_hash = hash_messages(prior_context)

    # state_hash: full conversation after appending assistant reply
    full_state = messages + [assistant_reply]
    state_hash = hash_messages(full_state)

    # tail_hash: last user message + assistant reply pair
    last_user = _find_last_user_msg(messages)
    tail_hash = _compute_tail_hash(last_user, assistant_reply) if last_user else _sha256(_canonical([assistant_reply]))

    # ── Two-pass parent lookup ────────────────────────────────────────
    parent_turn: Optional[ChatTurn] = None
    compacted = False

    if parent_hash:
        # Pass 1: strict full-context match
        parent_turn = await _pass1_strict(session, parent_hash)

        if parent_turn is None and prior_context:
            # Pass 2: tail anchor (compaction recovery)
            parent_turn = await _pass2_tail_anchor(
                session, prior_context, caller_hash,
            )
            if parent_turn is not None:
                compacted = True
                # Heal the chain: update parent's state_hash to the
                # compacted prior context so Pass 1 works next time
                await _heal_chain(session, parent_turn, parent_hash)

    # ── Determine conversation_id ─────────────────────────────────────
    if parent_turn is not None:
        conversation_id = parent_turn.conversation_id
    else:
        conversation_id = str(uuid.uuid4())

    # ── Extract token usage ───────────────────────────────────────────
    usage = usage or {}

    # ── Persist ───────────────────────────────────────────────────────
    turn = ChatTurn(
        conversation_id=conversation_id,
        parent_hash=parent_hash,
        state_hash=state_hash,
        tail_hash=tail_hash,
        caller_hash=caller_hash,
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
        compacted=compacted,
    )
    session.add(turn)
    await session.commit()

    if parent_turn:
        link_type = "compacted" if compacted else "linked"
        logger.debug(
            "chat_turn %s conversation=%s turn=%s parent=%s",
            link_type, conversation_id, turn.id, parent_turn.id,
        )
    else:
        logger.debug(
            "chat_turn new conversation=%s turn=%s",
            conversation_id, turn.id,
        )

    return turn
