"""API key lifecycle helpers.

API keys are opaque bearer tokens used for service-to-service auth
(e.g. agent worker → platform proxy). The key format is::

    sk-agent-<prefix>-<secret>

``prefix`` is stored in cleartext for O(1) lookup; only a SHA-256 hash
of the full key is persisted.  ``scopes`` (JSON list) grants narrow
permissions, e.g. ``["agent:<id>", "model:<id>"]``.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.settings import ApiKey, ApiKeyStatus


_PREFIX_LEN = 8
_KEY_NS = "sk-agent"


def agent_scope(agent_id: str) -> str:
    return f"agent:{agent_id}"


def model_scope(model_id: str) -> str:
    return f"model:{model_id}"


def _scoped(scope: str):
    # ``api_keys.scopes`` is ``json`` not ``jsonb``, so cast at query
    # time to use the ``?`` (array-element exists) operator.
    return sa.cast(ApiKey.scopes, JSONB).op("?")(scope)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _generate() -> tuple[str, str]:
    prefix = secrets.token_urlsafe(_PREFIX_LEN)[:_PREFIX_LEN]
    secret = secrets.token_urlsafe(48)
    return prefix, f"{_KEY_NS}-{prefix}-{secret}"


async def create_api_key(
    session: AsyncSession,
    *,
    name: str,
    scopes: list[str],
    created_by_id: str,
) -> tuple[ApiKey, str]:
    """Mint a new API key. Returns (ORM row, plaintext).

    The plaintext is only available here — never stored or logged.
    """
    prefix, raw = _generate()
    row = ApiKey(
        name=name,
        prefix=prefix,
        hashed_key=_hash(raw),
        scopes=scopes,
        status=ApiKeyStatus.active,
        created_by_id=created_by_id,
    )
    session.add(row)
    await session.flush()
    return row, raw


async def verify_api_key(session: AsyncSession, raw: str) -> Optional[ApiKey]:
    """Validate a raw API key.  Returns the active row or ``None``."""
    parts = raw.split("-", 3)
    if len(parts) != 4 or f"{parts[0]}-{parts[1]}" != _KEY_NS:
        return None
    prefix = parts[2]
    stmt = sa.select(ApiKey).where(
        ApiKey.prefix == prefix,
        ApiKey.status == ApiKeyStatus.active,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    if not secrets.compare_digest(row.hashed_key, _hash(raw)):
        return None
    return row


async def touch_last_used(session: AsyncSession, key_id: str) -> None:
    # ``api_keys.last_used_at`` is declared as ``TIMESTAMP WITHOUT TIME
    # ZONE``, so we must pass a naive UTC datetime — asyncpg refuses to
    # bind an offset-aware value against that column.
    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    await session.execute(
        sa.update(ApiKey)
        .where(ApiKey.id == key_id)
        .values(last_used_at=now_utc_naive)
    )


async def revoke_by_scope(session: AsyncSession, scope: str) -> int:
    """Revoke every active key whose ``scopes`` contains ``scope``."""
    result = await session.execute(
        sa.update(ApiKey)
        .where(ApiKey.status == ApiKeyStatus.active, _scoped(scope))
        .values(status=ApiKeyStatus.revoked)
    )
    return result.rowcount or 0


async def delete_by_scope(session: AsyncSession, scope: str) -> int:
    """Hard-delete every key whose ``scopes`` contains ``scope``.

    Use for agent-scoped keys that are being replaced (``start_agent``)
    or whose owning agent is going away (``delete_agent``).  Use
    :func:`revoke_by_scope` when you want to keep an audit trail.
    """
    result = await session.execute(
        sa.delete(ApiKey).where(_scoped(scope))
    )
    return result.rowcount or 0
