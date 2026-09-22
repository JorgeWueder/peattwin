"""Servicio de autenticacion: usuarios, credenciales y roles iniciales."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.role import Role
from app.models.user import User


def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def get_by_identifier(db: Session, identifier: str) -> User | None:
    """Busca por username o por email."""
    return db.scalars(
        select(User).where(
            (User.username == identifier) | (User.email == identifier)
        )
    ).first()


def count_users(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(User)) or 0)


def authenticate(db: Session, identifier: str, password: str) -> User | None:
    user = get_by_identifier(db, identifier)
    if user and verify_password(password, user.hashed_password):
        return user
    return None


def create_user(
    db: Session,
    *,
    email: str,
    username: str,
    password: str,
    full_name: str | None = None,
    role_names: list[str] | None = None,
    is_active: bool = True,
    is_superuser: bool = False,
) -> User:
    user = User(
        email=email,
        username=username,
        hashed_password=hash_password(password),
        full_name=full_name,
        is_active=is_active,
        is_superuser=is_superuser,
    )
    if role_names:
        user.roles = list(
            db.scalars(select(Role).where(Role.name.in_(role_names)))
        )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def touch_last_login(db: Session, user: User) -> None:
    user.last_login_at = func.now()
    db.commit()
    db.refresh(user)
