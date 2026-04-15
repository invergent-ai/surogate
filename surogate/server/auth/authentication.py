from datetime import datetime, timedelta, timezone
import secrets
from typing import Optional, Tuple, Tuple

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from surogate.core.db.engine import get_session
from surogate.core.db.repository import auth as repository
from surogate.core.db.models.platform import User

DEFAULT_ADMIN_USERNAME = "surogate"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()  # Reads Authorization: Bearer <token>


async def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    return await _get_current_subject(
        credentials,
        session,
        allow_password_change=False,
    )


async def require_model_access(
    model_id: str,
    credentials: HTTPAuthorizationCredentials,
    session: AsyncSession,
) -> str:
    """Accept a JWT (user) or an API key scoped to ``model:<model_id>``.

    Returns a subject identifier: the username for JWTs, or
    ``api-key:<key_id>`` for API keys.
    """
    from surogate.core.db.repository import api_keys as _api_keys_repo

    token = credentials.credentials
    if token.startswith("sk-agent-"):
        key = await _api_keys_repo.verify_api_key(session, token)
        if key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        from surogate.core.db.repository.api_keys import model_scope
        scope = model_scope(model_id)
        if not key.scopes or scope not in key.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key not scoped for this model",
            )
        await _api_keys_repo.touch_last_used(session, key.id)
        return f"api-key:{key.id}"

    return await _get_current_subject(
        credentials, session, allow_password_change=False,
    )


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
) -> str:
    """Validate JWT and return the user's UUID (not username)."""
    from surogate.core.db.models.platform import User
    username = await _get_current_subject(
        credentials, session, allow_password_change=False,
    )
    row = await session.execute(
        sa.select(User.id).where(User.username == username)
    )
    user_id = row.scalar_one_or_none()
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user_id


def _decode_subject_without_verification(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        )
    except jwt.InvalidTokenError:
        return None

    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None


async def _get_current_subject(
    credentials: HTTPAuthorizationCredentials,
    session: AsyncSession,
    *,
    allow_password_change: bool
) -> str:
    token = credentials.credentials
    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    record = await repository.get_user_and_secret(session, subject)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    _salt, _pwd_hash, jwt_secret, must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        if payload.get("sub") != subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        if must_change_password and not allow_password_change:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password change required",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def create_access_token(
    subject: str,
    session: AsyncSession,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Tokens are valid across restarts because the signing secret is stored in SQLite.
    """
    to_encode = {"sub": subject}
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        await _get_secret_for_subject(subject, session),
        algorithm = ALGORITHM,
    )
    
async def create_refresh_token(subject: str, session: AsyncSession) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs) and expire after REFRESH_TOKEN_EXPIRE_DAYS.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days = REFRESH_TOKEN_EXPIRE_DAYS)
    await repository.save_refresh_token(token, subject, expires_at.isoformat(), session)
    return token


async def refresh_access_token(
    refresh_token: str, session: AsyncSession
) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate a refresh token and issue a new access token.

    The refresh token itself is NOT consumed — it stays valid until expiry.
    Returns a new access_token or None if the refresh token is invalid/expired.
    """
    username = await repository.verify_refresh_token(refresh_token, session)
    if username is None:
        return None, None
    return await create_access_token(subject = username, session = session), username

async def get_current_subject_allow_password_change(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    return await _get_current_subject(
        credentials,
        session,
        allow_password_change = True,
    )
    
async def verify_ws_token(token: str, session: AsyncSession) -> Optional[str]:
    """Validate a JWT for WebSocket auth. Returns the subject or None."""
    subject = _decode_subject_without_verification(token)
    if subject is None:
        return None
    record = await repository.get_user_and_secret(session, subject)
    if record is None:
        return None
    _salt, _pwd_hash, jwt_secret, _must_change = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        if payload.get("sub") != subject:
            return None
        return subject
    except jwt.InvalidTokenError:
        return None


async def _get_secret_for_subject(subject: str, session: AsyncSession) -> str:
    secret = await repository.get_jwt_secret(subject, session)
    if secret is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired token",
        )
    return secret




