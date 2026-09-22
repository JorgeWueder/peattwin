"""Rol de acceso (admin, investigador, visualizador...)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.associations import role_permissions, user_roles
from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.permission import Permission
    from app.models.user import User


class Role(Base, TimestampMixin):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    users: Mapped[list["User"]] = relationship(
        secondary=user_roles, back_populates="roles"
    )
    permissions: Mapped[list["Permission"]] = relationship(
        secondary=role_permissions, back_populates="roles", lazy="selectin"
    )
