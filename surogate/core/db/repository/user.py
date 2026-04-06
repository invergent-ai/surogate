from typing import Optional
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.platform import (
    User,
)

async def get_user_by_username(session: AsyncSession, username: str) -> Optional[User]:
    result = await session.execute(
        sa.select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()

async def get_lakefs_credentials(session: AsyncSession, user_identifier: str) -> Optional[tuple[str, str]]:
    """Return LakeFS credentials (key, secret) for a user.

    *user_identifier* can be either a username or a user UUID.
    """
    # Try by username first (short strings), then by id (UUID format)
    result = await session.execute(
        sa.select(User.hub_key, User.hub_secret)
        .where(sa.or_(User.username == user_identifier, User.id == user_identifier))
    )
    return result.one_or_none()

async def set_lakefs_credentials(
    session: AsyncSession, username: str, key: str, secret: str
) -> None:
    """Set LakeFS credentials (key, secret) for a user."""
    await session.execute(
        sa.update(User)
        .where(User.username == username)
        .values(hub_key=key, hub_secret=secret)
    )
    await session.commit()