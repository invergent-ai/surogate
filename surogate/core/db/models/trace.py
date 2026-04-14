"""Trace domain: ChatTurn – records every proxied chat-completion round-trip."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from surogate.core.db.base import Base, UUIDMixin


class ChatTurn(UUIDMixin, Base):
    """One request/response round-trip through the chat-completion proxy."""

    __tablename__ = "chat_turns"

    # ── Conversation grouping ─────────────────────────────────────────
    conversation_id: Mapped[str] = mapped_column(
        sa.String(36), index=True,
        doc="Shared by every turn in the same conversation tree.",
    )
    parent_hash: Mapped[Optional[str]] = mapped_column(
        sa.String(64), index=True, nullable=True,
        doc="sha256 of messages[:-1]; NULL for root turns.",
    )
    state_hash: Mapped[str] = mapped_column(
        sa.String(64), index=True,
        doc="sha256 of full message array including assistant reply.",
    )
    tail_hash: Mapped[str] = mapped_column(
        sa.String(64), index=True,
        doc="sha256 of [last_user_msg, assistant_reply] pair.",
    )

    # ── Caller identity ───────────────────────────────────────────────
    caller_hash: Mapped[Optional[str]] = mapped_column(
        sa.String(64), nullable=True, index=True,
        doc="sha256 of Authorization token – identifies the caller.",
    )

    # ── Request / response metadata ──────────────────────────────────
    deployed_model_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), index=True, nullable=True,
        doc="FK-style pointer to the DeployedModel that served this turn.",
    )
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
