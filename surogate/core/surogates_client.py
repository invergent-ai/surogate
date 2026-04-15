"""Client for the embedded Surogates database.

A single long-lived process-wide client owns the engine + session
factory for the ``surogates`` database.  All surogate-server code that
needs to read or mutate surogates data (orgs, skills, tools, audit
records, …) should go through this client so we have one connection
pool, one place to add tracing, and consistent use of the
``surogates.*`` ORM models and Pydantic types.

Typical usage::

    surogates = request.app.state.surogates
    org = await surogates.ensure_org(project_id, project_name)
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession

from surogates.config import DatabaseSettings
from surogates.db.engine import async_engine_from_settings, async_session_factory
from surogates.db.models import (
    DeliveryOutbox,
    Event,
    Org,
    Session as SessionRow,
    SessionCursor,
    SessionLease,
    User,
)
from surogates.tenant.auth.database import DatabaseAuthProvider

from surogate.utils.logger import get_logger

logger = get_logger()


class SurogatesClient:
    """Process-wide handle to the surogates database.

    Holds an :class:`AsyncEngine` and a session factory.  Dispose via
    :meth:`close` on shutdown.
    """

    def __init__(self, database_url: str) -> None:
        self._settings = DatabaseSettings(url=database_url)
        self._engine: AsyncEngine = async_engine_from_settings(self._settings)
        self._session_factory: async_sessionmaker[AsyncSession] = (
            async_session_factory(self._engine)
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def close(self) -> None:
        await self._engine.dispose()

    # ── Orgs ─────────────────────────────────────────────────────────

    async def ensure_org(self, org_id: UUID, name: str) -> Org:
        """Idempotently create the org row for a surogate project.

        Uses ``ON CONFLICT (id) DO NOTHING`` so concurrent callers are
        safe.  Returns the row (existing or newly created).
        """
        async with self._session_factory() as session:
            stmt = (
                pg_insert(Org)
                .values(id=org_id, name=name)
                .on_conflict_do_nothing(index_elements=[Org.id])
            )
            await session.execute(stmt)
            await session.commit()

            row = (
                await session.execute(select(Org).where(Org.id == org_id))
            ).scalar_one()
            return row

    async def get_org(self, org_id: UUID) -> Optional[Org]:
        async with self._session_factory() as session:
            return (
                await session.execute(select(Org).where(Org.id == org_id))
            ).scalar_one_or_none()

    # ── Users ────────────────────────────────────────────────────────

    async def list_users(self, org_id: UUID) -> list[User]:
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(User).where(User.org_id == org_id).order_by(User.created_at)
                )
            ).scalars().all()
            return list(rows)

    async def create_user(
        self,
        *,
        org_id: UUID,
        email: str,
        display_name: str,
        password: Optional[str],
    ) -> User:
        """Create a user.  When ``password`` is given the user can log in
        via the ``database`` provider; otherwise the user is treated as
        externally-authenticated (SSO) with no local password.
        """
        password_hash = (
            DatabaseAuthProvider.hash_password(password) if password else None
        )
        async with self._session_factory() as session:
            user = User(
                org_id=org_id,
                email=email,
                display_name=display_name,
                auth_provider="database" if password else "external",
                password_hash=password_hash,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def update_user(
        self,
        user_id: UUID,
        *,
        display_name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Optional[User]:
        async with self._session_factory() as session:
            user = (
                await session.execute(select(User).where(User.id == user_id))
            ).scalar_one_or_none()
            if user is None:
                return None
            if display_name is not None:
                user.display_name = display_name
            if password is not None:
                user.password_hash = DatabaseAuthProvider.hash_password(password)
                user.auth_provider = "database"
            await session.commit()
            await session.refresh(user)
            return user

    # ── Agent data scrub ─────────────────────────────────────────────

    async def delete_agent_data(self, agent_id: str) -> int:
        """Hard-delete every session (and downstream rows) for an agent.

        ``sessions.agent_id`` is a plain ``Text`` column — not an FK —
        so we delete by scoped ID.  All five session-child tables are
        cleared in one transaction before the sessions themselves.

        Returns the number of sessions deleted.
        """
        async with self._session_factory() as session:
            session_ids_subq = (
                select(SessionRow.id).where(SessionRow.agent_id == agent_id)
            )
            await session.execute(
                delete(Event).where(Event.session_id.in_(session_ids_subq))
            )
            await session.execute(
                delete(SessionLease).where(
                    SessionLease.session_id.in_(session_ids_subq)
                )
            )
            await session.execute(
                delete(SessionCursor).where(
                    SessionCursor.session_id.in_(session_ids_subq)
                )
            )
            await session.execute(
                delete(DeliveryOutbox).where(
                    DeliveryOutbox.session_id.in_(session_ids_subq)
                )
            )
            # Break the self-referencing ``sessions.parent_id`` link
            # before deleting the rows.  PostgreSQL checks FKs per row
            # so deleting a chain of parent/child sessions in one
            # statement can fail even when every row in the chain is
            # going away.
            await session.execute(
                update(SessionRow)
                .where(SessionRow.agent_id == agent_id)
                .values(parent_id=None)
            )
            result = await session.execute(
                delete(SessionRow).where(SessionRow.agent_id == agent_id)
            )
            await session.commit()
            return result.rowcount or 0

    async def delete_user(self, user_id: UUID) -> bool:
        async with self._session_factory() as session:
            user = (
                await session.execute(select(User).where(User.id == user_id))
            ).scalar_one_or_none()
            if user is None:
                return False
            await session.delete(user)
            await session.commit()
            return True
