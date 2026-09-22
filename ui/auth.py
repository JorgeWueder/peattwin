"""Sesion autenticada de Streamlit: login, RBAC y logout.

La sesion vive en `st.session_state`. Al entrar se toma un SNAPSHOT inmutable
del usuario (id, nombre, roles y codigos de permiso) mientras la sesion de
SQLAlchemy sigue abierta; nunca se guarda la entidad ORM, que quedaria
desprendida en el siguiente rerun.

Nota: refrescar el navegador (F5) crea una sesion nueva de Streamlit y obliga a
volver a entrar. Es el comportamiento estandar del framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from app.services import access, auth_service
from ui.db import session_scope

_KEY = "peattwin_user"

ROLE_ORDER = ("admin", "investigador", "visualizador")


@dataclass(frozen=True)
class SessionUser:
    """Copia plana del usuario autenticado, segura entre reruns."""

    id: int
    username: str
    email: str
    full_name: str | None
    is_superuser: bool
    role_names: tuple[str, ...] = ()
    permission_codes: frozenset[str] = field(default_factory=frozenset)

    @property
    def display_name(self) -> str:
        return self.full_name or self.username

    @property
    def primary_role(self) -> str:
        for role in ROLE_ORDER:
            if role in self.role_names:
                return role
        return self.role_names[0] if self.role_names else "sin rol"


def _snapshot(user) -> SessionUser:
    """Requiere una sesion abierta: recorre user.roles y role.permissions."""
    names = sorted(access.role_names(user), key=lambda r: (ROLE_ORDER.index(r) if r in ROLE_ORDER else 99, r))
    return SessionUser(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_superuser=bool(user.is_superuser),
        role_names=tuple(names),
        permission_codes=frozenset(access.permission_codes(user)),
    )


def login(identifier: str, password: str) -> tuple[bool, str | None]:
    """Autentica y abre sesion. Devuelve (ok, mensaje_de_error)."""
    with session_scope() as db:
        user = auth_service.authenticate(db, identifier.strip(), password)
        if user is None:
            return False, "Usuario o contrasena incorrectos."
        if not user.is_active:
            return False, "Usuario inactivo. Pide a un administrador que lo reactive."
        auth_service.touch_last_login(db, user)
        st.session_state[_KEY] = _snapshot(user)
    return True, None


def logout() -> None:
    """Cierra la sesion barriendo TODO `st.session_state`, no solo el usuario.

    Cada clave de la app pertenece al usuario que cierra sesion: el nivel
    freatico del simulador, el escenario guardado, los bytes de los informes ya
    generados (`report_*`, `_scenario_docx`) y el estado de los widgets del
    panel de administracion (`roles_*`, `active_*`). Si solo se borrase la clave
    del usuario, al entrar con otra cuenta seguirian visibles los botones de
    descarga del anterior. El `st.rerun()` lo hace el llamador.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()


def current_user() -> SessionUser | None:
    return st.session_state.get(_KEY)


def is_authenticated() -> bool:
    return _KEY in st.session_state


def refresh_session() -> None:
    """Recarga roles y permisos desde la BD (tras editarse en administracion)."""
    user = current_user()
    if user is None:
        return
    with session_scope() as db:
        fresh = auth_service.get_user(db, user.id)
        if fresh is None or not fresh.is_active:
            logout()
            return
        st.session_state[_KEY] = _snapshot(fresh)


def has_role(*roles: str) -> bool:
    user = current_user()
    if user is None:
        return False
    return user.is_superuser or bool(set(user.role_names) & set(roles))


def has_permission(*codes: str) -> bool:
    """Misma semantica que el RBAC anterior: superuser y admin lo superan todo."""
    user = current_user()
    if user is None:
        return False
    return (
        user.is_superuser
        or "admin" in user.role_names
        or bool(user.permission_codes & set(codes))
    )
