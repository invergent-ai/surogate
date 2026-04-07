"""Trace domain: ChatTurn – hash-linked conversation tracking for proxied LLM calls."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from surogate.core.db.base import Base, UUIDMixin


class ChatTurn(UUIDMixin, Base):
    """One request/response round-trip through the chat-completion proxy.

    Turns are linked into a conversation DAG using dual-key indexing:

    **Pass 1 – Strict match (no compaction):**

    *   **parent_hash** – ``sha256(messages[:-1])``.  Matches a prior turn's
        ``state_hash``, linking the two.  ``NULL`` for conversation roots.
    *   **state_hash** – ``sha256(messages + [assistant_reply])``.
        Becomes the ``parent_hash`` of the *next* turn.

    **Pass 2 – Tail anchor (compaction recovery):**

    *   **tail_hash** – ``sha256([last_user_msg, assistant_reply])``.
        When the agent compacts older messages the full-context chain
        breaks, but the most recent user+assistant pair is almost always
        preserved.  Matching on this pair recovers the thread.

    After a compaction match the proxy *heals* the chain by updating the
    matched turn's ``state_hash`` so that subsequent Pass-1 lookups
    succeed without hitting Pass 2 again.

    Branching (regenerations, A/B tests) is free: two turns with the same
    ``parent_hash`` but different responses simply create two children.
    """

    __tablename__ = "chat_turns"

    # ── Conversation linkage ──────────────────────────────────────────
    conversation_id: Mapped[str] = mapped_column(
        sa.String(36), index=True,
        doc="Shared by every turn in the same conversation tree.",
    )
    parent_hash: Mapped[Optional[str]] = mapped_column(
        sa.String(64), index=True, nullable=True,
        doc="sha256 of messages[:-1]; NULL for root turns.",
    )
    state_hash: Mapped[str] = mapped_column(
        sa.String(64), index=True, unique=True,
        doc="sha256 of full message array including assistant reply.",
    )
    tail_hash: Mapped[str] = mapped_column(
        sa.String(64), index=True,
        doc="sha256 of [last_user_msg, assistant_reply] pair (compaction fallback).",
    )

    # ── Caller identity ───────────────────────────────────────────────
    caller_hash: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True, index=True,
        doc="sha256 of Authorization token – identifies the caller.",
    )

    # ── Request / response metadata ──────────────────────────────────
    project_name: Mapped[str] = mapped_column(sa.String(255))
    run_name: Mapped[str] = mapped_column(sa.String(255))
    model: Mapped[str] = mapped_column(sa.String(255), default="")
    is_streaming: Mapped[bool] = mapped_column(sa.Boolean, default=False)

    prompt_tokens: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)

    latency_ms: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    request_body: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True,
        doc="Full chat-completion request payload.",
    )
    response_body: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True,
        doc="Full chat-completion response payload (non-streaming only).",
    )

    compacted: Mapped[bool] = mapped_column(
        sa.Boolean, default=False,
        doc="True when linked via tail_hash fallback (history was compacted).",
    )

    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now(),
    )


__all__ = ["ChatTurn"]
