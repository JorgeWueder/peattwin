"""Usuarios y roles. Vista exclusiva del rol `admin`.

Los roles y permisos se siembran en la migracion `0002_digital_twin_schema`;
aqui se listan como referencia y se asignan a los usuarios.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.schemas.auth import UserAdminUpdate
from app.services import access, rbac_service
from ui import auth
from ui import format as F
from ui import interpret as I
from ui.db import session_scope


def _load() -> tuple[list[dict], list[dict], list[dict]]:
    """Snapshot plano de usuarios, roles y permisos (fuera de la sesion ORM)."""
    with session_scope() as db:
        users = [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "full_name": u.full_name,
                "is_active": u.is_active,
                "is_superuser": u.is_superuser,
                "last_login_at": u.last_login_at,
                "role_ids": [r.id for r in u.roles],
                "role_names": sorted(access.role_names(u)),
            }
            for u in rbac_service.list_users(db)
        ]
        roles = [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "permissions": sorted(p.code for p in r.permissions),
            }
            for r in rbac_service.list_roles(db)
        ]
        permissions = [
            {"code": p.code, "description": p.description}
            for p in rbac_service.list_permissions(db)
        ]
    return users, roles, permissions


def _save(user_id: int, role_ids: list[int], is_active: bool) -> None:
    with session_scope() as db:
        rbac_service.update_user(
            db, user_id, UserAdminUpdate(is_active=is_active, role_ids=role_ids)
        )


# ---------------------------------------------------------------- lecturas
# Tambien aqui: la regla del proyecto es que ninguna tabla se muestre sin decir
# que hay que mirar en ella. En un panel de administracion lo relevante no es el
# contenido de una fila sino el estado agregado del control de acceso, que es lo
# que puede estar mal sin que salte ningun error.

def _lectura_usuarios(users: list[dict]) -> str:
    total = len(users)
    activos = sum(1 for u in users if u["is_active"])
    sin_rol = [u["username"] for u in users if not u["role_names"]]
    superusers = [u["username"] for u in users if u["is_superuser"]]
    admins = [u["username"] for u in users if "admin" in u["role_names"]]
    nunca = sum(1 for u in users if not u["last_login_at"])

    avisos = []
    if sin_rol:
        avisos.append(
            f"{len(sin_rol)} cuenta(s) sin ningun rol (@{', @'.join(sin_rol)}): pueden "
            "iniciar sesion pero no veran mas que las vistas publicas, porque toda la "
            "navegacion se filtra por permisos."
        )
    if len(admins) <= 1:
        avisos.append(
            "Solo hay una cuenta con rol admin: si se desactiva o se pierde su clave, "
            "nadie podra volver a asignar roles desde la interfaz."
        )
    if len(superusers) > 1:
        avisos.append(
            f"Hay {len(superusers)} superusers (@{', @'.join(superusers)}); el flag "
            "salta el chequeo de permisos, asi que conviene que sea excepcional."
        )
    cola = " " + " ".join(avisos) if avisos else (
        " No hay cuentas huerfanas de rol ni superusers de mas."
    )
    return (
        f"{total} cuenta(s), {activos} activa(s) y {total - activos} desactivada(s); "
        f"{nunca} no ha iniciado sesion nunca. El acceso efectivo no lo da esta tabla sino "
        "la columna Roles: los permisos se heredan del rol, nunca se asignan a la persona, "
        f"y el flag Superuser se salta esa comprobacion entera.{cola}"
    )


def _lectura_permisos(permissions: list[dict], roles: list[dict]) -> str:
    usados = {c for r in roles for c in r["permissions"]}
    huerfanos = sorted({p["code"] for p in permissions} - usados)
    cola = (
        f" {len(huerfanos)} permiso(s) no estan en ningun rol ({', '.join(huerfanos)}): "
        "estan definidos pero hoy nadie los tiene, asi que la funcionalidad que protegen "
        "esta cerrada para todo el mundo."
        if huerfanos
        else " Todos los permisos estan asignados a algun rol; no hay codigos muertos."
    )
    return (
        f"Los {len(permissions)} codigos de permiso que entiende la aplicacion, sembrados "
        "en la migracion 0002. Se leen en pareja con la columna de Roles de la izquierda: "
        "esta tabla dice que existe y el panel de roles dice quien lo tiene. El formato "
        "`recurso:accion` es el que comprueban `auth.has_permission()` en las vistas y "
        f"`rbac_service` en la capa de servicio, con doble guarda.{cola}"
    )


def _restricted_notice() -> None:
    user = auth.current_user()
    roles = ", ".join(user.role_names) if user and user.role_names else "sin rol"
    st.warning(
        f"**Acceso restringido.** Esta vista es exclusiva del rol **admin**; tu sesion "
        f"actual es **{roles}**. Con rol **investigador** puedes gestionar escenarios y "
        "ejecutar el simulador; con **visualizador**, consultar series y modelos."
    )


def render() -> None:
    st.title("Usuarios y roles")

    # Doble guarda: la pagina no se registra en la navegacion sin rol admin,
    # y ademas se comprueba aqui.
    if not auth.has_role("admin"):
        _restricted_notice()
        return

    st.caption(
        "Alta de usuarios, asignacion de roles y activacion. Los roles y permisos se "
        "siembran en la migracion 0002; aqui se listan como referencia."
    )

    users, roles, permissions = _load()
    role_by_id = {r["id"]: r["name"] for r in roles}
    current = auth.current_user()

    I.titulo("Usuarios")
    if not users:
        st.info("No hay usuarios.")
        return

    I.tabla(
        pd.DataFrame(
            [
                {
                    "Usuario": u["full_name"] or u["username"],
                    "Cuenta": f"@{u['username']}",
                    "Email": u["email"],
                    "Roles": ", ".join(u["role_names"]) or "—",
                    "Estado": "activo" if u["is_active"] else "inactivo",
                    "Superuser": "si" if u["is_superuser"] else "",
                    "Ultimo acceso": F.fmt_date(u["last_login_at"]),
                }
                for u in users
            ]
        ),
        explicabilidad=(
            "Todas las cuentas de la tabla `users`. «Roles» lista los roles asignados, "
            "que son los que conceden permisos; «Estado» distingue una cuenta activa de "
            "una desactivada, que existe pero no puede iniciar sesion; «Superuser» es un "
            "flag que salta la comprobacion de permisos por completo. «Ultimo acceso» "
            "queda vacio si la cuenta nunca ha entrado."
        ),
        interpretacion=_lectura_usuarios(users),
    )

    st.subheader("Editar un usuario")
    target = st.selectbox(
        "Usuario",
        users,
        format_func=lambda u: f"{u['full_name'] or u['username']} (@{u['username']})"
        + (" · tu sesion" if u["id"] == current.id else ""),
    )
    # Las claves llevan el id del usuario: sin ellas Streamlit reutilizaria el
    # estado del widget al cambiar de usuario y mostraria los roles del anterior.
    with st.form(f"edit_user_{target['id']}"):
        chosen = st.multiselect(
            "Roles",
            options=list(role_by_id),
            default=target["role_ids"],
            format_func=lambda rid: role_by_id[rid],
            key=f"roles_{target['id']}",
        )
        is_active = st.checkbox(
            "Cuenta activa", value=target["is_active"], key=f"active_{target['id']}"
        )
        submitted = st.form_submit_button("Guardar cambios", type="primary")

    if submitted:
        if target["id"] == current.id and not is_active:
            st.error("No puedes desactivar tu propia cuenta.")
        elif target["id"] == current.id and "admin" not in {role_by_id[r] for r in chosen}:
            st.error("No puedes quitarte a ti mismo el rol admin: te quedarias sin acceso.")
        else:
            try:
                _save(target["id"], chosen, is_active)
            except Exception as exc:
                st.error(f"No se pudo guardar: {exc}")
            else:
                if target["id"] == current.id:
                    auth.refresh_session()
                st.cache_data.clear()
                st.success(f"Usuario @{target['username']} actualizado.")
                st.rerun()

    st.divider()
    col_roles, col_perms = st.columns(2)

    with col_roles:
        st.subheader("Roles")
        st.caption("Sembrados en la migracion 0002.")
        for r in roles:
            with st.container(border=True):
                st.markdown(f"**{r['name']}** · {len(r['permissions'])} permisos")
                if r["description"]:
                    st.caption(r["description"])
                st.code(" ".join(r["permissions"]) or "—", language=None)

    with col_perms:
        I.titulo("Permisos")
        I.tabla(
            pd.DataFrame(
                [{"Codigo": p["code"], "Descripcion": p["description"]} for p in permissions]
            ),
            explicabilidad=(
                "Catalogo de codigos de permiso que reconoce la aplicacion, sembrados "
                "por la migracion 0002. El formato es `recurso:accion`. Esta tabla dice "
                "que permisos EXISTEN; el panel de Roles de la izquierda dice quien los "
                "tiene."
            ),
            interpretacion=_lectura_permisos(permissions, roles),
        )
