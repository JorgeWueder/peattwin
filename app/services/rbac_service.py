"""CRUD de roles, permisos y asignaciones de usuario (modulo admin)."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.permission import Permission
from app.models.role import Role
from app.models.user import User
from app.schemas.auth import (
    PermissionCreate,
    PermissionUpdate,
    RoleCreate,
    RoleUpdate,
    UserAdminUpdate,
)


# ------------------------------------------------------------------ permissions
def list_permissions(db: Session) -> list[Permission]:
    return list(db.scalars(select(Permission).order_by(Permission.code)))


def get_permission(db: Session, permission_id: int) -> Permission | None:
    return db.get(Permission, permission_id)


def create_permission(db: Session, payload: PermissionCreate) -> Permission:
    perm = Permission(**payload.model_dump())
    db.add(perm)
    db.commit()
    db.refresh(perm)
    return perm


def update_permission(
    db: Session, permission_id: int, payload: PermissionUpdate
) -> Permission | None:
    perm = db.get(Permission, permission_id)
    if perm is None:
        return None
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(perm, key, value)
    db.commit()
    db.refresh(perm)
    return perm


def delete_permission(db: Session, permission_id: int) -> bool:
    perm = db.get(Permission, permission_id)
    if perm is None:
        return False
    db.delete(perm)
    db.commit()
    return True


# ------------------------------------------------------------------------ roles
def list_roles(db: Session) -> list[Role]:
    return list(db.scalars(select(Role).order_by(Role.name)))


def get_role(db: Session, role_id: int) -> Role | None:
    return db.get(Role, role_id)


def _load_permissions(db: Session, ids: list[int]) -> list[Permission]:
    if not ids:
        return []
    return list(db.scalars(select(Permission).where(Permission.id.in_(ids))))


def create_role(db: Session, payload: RoleCreate) -> Role:
    role = Role(name=payload.name, description=payload.description)
    role.permissions = _load_permissions(db, payload.permission_ids)
    db.add(role)
    db.commit()
    db.refresh(role)
    return role


def update_role(db: Session, role_id: int, payload: RoleUpdate) -> Role | None:
    role = db.get(Role, role_id)
    if role is None:
        return None
    data = payload.model_dump(exclude_unset=True)
    if "name" in data:
        role.name = data["name"]
    if "description" in data:
        role.description = data["description"]
    if data.get("permission_ids") is not None:
        role.permissions = _load_permissions(db, data["permission_ids"])
    db.commit()
    db.refresh(role)
    return role


def delete_role(db: Session, role_id: int) -> bool:
    role = db.get(Role, role_id)
    if role is None:
        return False
    db.delete(role)
    db.commit()
    return True


# ------------------------------------------------------------------------ users
def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.id)))


def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def update_user(
    db: Session, user_id: int, payload: UserAdminUpdate
) -> User | None:
    user = db.get(User, user_id)
    if user is None:
        return None
    data = payload.model_dump(exclude_unset=True)
    for field in ("full_name", "is_active", "is_superuser"):
        if field in data:
            setattr(user, field, data[field])
    if data.get("password"):
        user.hashed_password = hash_password(data["password"])
    if data.get("role_ids") is not None:
        user.roles = list(
            db.scalars(select(Role).where(Role.id.in_(data["role_ids"])))
        )
    db.commit()
    db.refresh(user)
    return user
