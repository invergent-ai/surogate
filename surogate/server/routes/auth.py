"""
Authentication API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.server.auth.authentication import (
    create_access_token,
    create_refresh_token,
    get_current_subject_allow_password_change,
    refresh_access_token,
)
from surogate.core.hub.lakefs import seed_lakefs_user
from surogate.core.db.engine import get_session
from surogate.core.db.repository import auth as auth_repository
from surogate.core.db.repository import project as project_repository
from surogate.server.models.auth import AuthLoginRequest, ChangePasswordRequest, RefreshTokenRequest, Token

from surogate.utils import hashing

router = APIRouter()

@router.post("/login", response_model = Token)
async def login(
    payload: AuthLoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Token:
    """
    Login with username/password and receive access + refresh tokens.
    """
    record = await auth_repository.get_user_and_secret(session, payload.username)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password.",
        )

    salt, pwd_hash, _jwt_secret, must_change_password = record
    
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect password.",
        )

    await project_repository.seed_default_project(session, payload.username)
    await seed_lakefs_user(payload.username, session, request.app.state.config)

    # Ensure dstack projects + K8s backends exist for all user projects
    from surogate.core.compute import ensure_dstack_project
    projects = await project_repository.get_user_projects(session, payload.username)
    for proj in projects:
        if not proj.dstack_project_name:
            try:
                dstack_proj = await ensure_dstack_project(proj.namespace)
                proj.dstack_project_name = dstack_proj.name
                await session.commit()
            except Exception:
                pass

    access_token = await create_access_token(payload.username, session)
    refresh_token = await create_refresh_token(payload.username, session)

    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = must_change_password,
    )
    
@router.post("/refresh", response_model = Token)
async def refresh(
    payload: RefreshTokenRequest,
    session: AsyncSession = Depends(get_session),
) -> Token:
    """
    Exchange a valid refresh token for a new access token.

    The refresh token itself is reusable until it expires (7 days).
    """
    new_access_token, username = await refresh_access_token(payload.refresh_token, session)
    if new_access_token is None or username is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired refresh token",
        )

    must_change_password = await auth_repository.requires_password_change(session, username)
    
    return Token(
        access_token = new_access_token,
        refresh_token = payload.refresh_token,
        token_type = "bearer",
        must_change_password = must_change_password,
    )
    
@router.post("/change-password", response_model = Token)
async def change_password(
    payload: ChangePasswordRequest,
    current_subject: str = Depends(get_current_subject_allow_password_change),
    session: AsyncSession = Depends(get_session),
) -> Token:
    """Allow the authenticated user to replace the default password."""
    record = await auth_repository.get_user_and_secret(session, current_subject)
    if record is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "User session is invalid",
        )

    salt, pwd_hash, _jwt_secret, _must_change_password = record
    if not hashing.verify_password(payload.current_password, salt, pwd_hash):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Current password is incorrect",
        )
    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "New password must be different from the current password",
        )

    await auth_repository.update_password(session, current_subject, payload.new_password)
    await auth_repository.revoke_user_refresh_tokens(session, current_subject)
    
    access_token = await create_access_token(current_subject, session)
    refresh_token = await create_refresh_token(current_subject, session)
    return Token(
        access_token = access_token,
        refresh_token = refresh_token,
        token_type = "bearer",
        must_change_password = False,
    )