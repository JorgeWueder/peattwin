"""Comprobaciones de acceso por rol y por permiso (RBAC).

Misma semantica que las dependencias de FastAPI que existian en `app/api/deps.py`:

  * `is_superuser` supera cualquier comprobacion.
  * En permisos, el rol `admin` tambien los supera todos.

Se llaman con el usuario ya cargado, dentro de una sesion abierta (recorren las
relaciones `user.roles` y `role.permissions`).
"""
from __future__ import annotations

from app.models.user import User


def role_names(user: User) -> set[str]:
    return {r.name for r in user.roles}


def permission_codes(user: User) -> set[str]:
    return {p.code for r in user.roles for p in r.permissions}


def has_role(user: User, *roles: str) -> bool:
    return bool(user.is_superuser or (role_names(user) & set(roles)))


def has_permission(user: User, *codes: str) -> bool:
    return bool(
        user.is_superuser
        or "admin" in role_names(user)
        or (permission_codes(user) & set(codes))
    )
