"""Platform core: User, Project, ProjectMember."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from surogate.core.db.base import Base, UUIDMixin, TimestampMixin


class UserRole(enum.Enum):
    admin = "admin"
    skill_author = "skill_author"
    devops = "devops"
    developer = "developer"

class ProjectStatus(enum.Enum):
    active = "active"
    archived = "archived"


class ProjectMemberRole(enum.Enum):
    owner = "owner"
    editor = "editor"
    viewer = "viewer"


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(sa.String(150), unique=True, index=True)
    name: Mapped[str] = mapped_column(sa.String(255))
    email: Mapped[Optional[str]] = mapped_column(sa.String(320), unique=True, nullable=True)
    salt: Mapped[str] = mapped_column(sa.String(64), default="")
    password_hash: Mapped[str] = mapped_column(sa.String(512), default="")
    jwt_secret: Mapped[str] = mapped_column(sa.String(256), default="")
    must_change_password: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    hub_key: Mapped[str] = mapped_column(sa.String(128), default="")
    hub_secret: Mapped[str] = mapped_column(sa.String(128), default="")
    avatar_url: Mapped[Optional[str]] = mapped_column(sa.String(2048), nullable=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    default_project_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("projects.id"), nullable=True
    )

    role_assignments: Mapped[list[UserRoleAssignment]] = relationship(
        back_populates="user", cascade="all, delete-orphan",
    )
    memberships: Mapped[list[ProjectMember]] = relationship(
        back_populates="user", foreign_keys="[ProjectMember.user_id]"
    )

    @property
    def roles(self) -> list[UserRole]:
        return [a.role for a in self.role_assignments]

    def has_role(self, role: UserRole) -> bool:
        return role in self.roles


class UserRoleAssignment(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), primary_key=True
    )
    role: Mapped[UserRole] = mapped_column(
        sa.Enum(UserRole), primary_key=True
    )

    user: Mapped[User] = relationship(back_populates="role_assignments")


class Project(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(sa.String(255))
    namespace: Mapped[str] = mapped_column(sa.String(255), unique=True, index=True)
    color: Mapped[str] = mapped_column(sa.String(32))
    status: Mapped[ProjectStatus] = mapped_column(
        sa.Enum(ProjectStatus), default=ProjectStatus.active
    )
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))
    dstack_project_name: Mapped[Optional[str]] = mapped_column(
        sa.String(50), nullable=True, unique=True
    )

    created_by: Mapped[User] = relationship(foreign_keys=[created_by_id])
    members: Mapped[list[ProjectMember]] = relationship(back_populates="project")


class ProjectMember(Base):
    __tablename__ = "project_members"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), primary_key=True
    )
    role: Mapped[ProjectMemberRole] = mapped_column(sa.Enum(ProjectMemberRole))
    added_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    added_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    project: Mapped[Project] = relationship(back_populates="members")
    user: Mapped[User] = relationship(
        back_populates="memberships", foreign_keys=[user_id]
    )
    added_by: Mapped[User] = relationship(foreign_keys=[added_by_id])


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    token_hash: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    username: Mapped[str] = mapped_column(sa.String(320), nullable=False, index=True)
    expires_at: Mapped[str] = mapped_column(sa.String(64), nullable=False)


__all__ = [
    "UserRole",
    "UserRoleAssignment",
    "ProjectStatus",
    "ProjectMemberRole",
    "User",
    "Project",
    "ProjectMember",
    "RefreshToken",
]
