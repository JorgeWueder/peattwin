"""Pantalla de acceso: entrar y crear cuenta.

El registro replica la regla que tenia la API: el PRIMER usuario del sistema
recibe el rol `admin` y `is_superuser`; el resto entran como `visualizador`.
"""
from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import IntegrityError

from app.services import auth_service
from ui import auth, theme
from ui.db import session_scope


def _register(email: str, username: str, password: str, full_name: str) -> tuple[bool, str]:
    with session_scope() as db:
        first_user = auth_service.count_users(db) == 0
        role_names = ["admin"] if first_user else ["visualizador"]
        try:
            auth_service.create_user(
                db,
                email=email.strip(),
                username=username.strip(),
                password=password,
                full_name=full_name.strip() or None,
                role_names=role_names,
                is_superuser=first_user,
            )
        except IntegrityError:
            db.rollback()
            return False, "El email o el nombre de usuario ya estan registrados."
    rol = "admin" if first_user else "visualizador"
    return True, f"Cuenta creada con rol «{rol}». Ya puedes entrar."


def render() -> None:
    # Solo estilos: el ancho de dashboard deja este formulario estirado.
    theme.inject_login()

    st.markdown("## PeatTwin")
    st.caption(
        "Gemelo Digital de Turberas — restauracion hidrologica y balance de "
        "carbono CO₂–CH₄ del sitio DE-Zrk (Zarnekow)."
    )

    entrar, crear = st.tabs(["Entrar", "Crear una cuenta"])

    with entrar:
        with st.form("login_form"):
            identifier = st.text_input("Usuario o email", autocomplete="username")
            password = st.text_input("Contrasena", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Entrar", type="primary")
        if submitted:
            if not identifier or not password:
                st.error("Introduce usuario y contrasena.")
            else:
                ok, err = auth.login(identifier, password)
                if ok:
                    st.rerun()
                else:
                    st.error(err)

    with crear:
        with session_scope() as db:
            sera_admin = auth_service.count_users(db) == 0
        if sera_admin:
            st.info(
                "No hay ningun usuario todavia: esta primera cuenta sera el "
                "**administrador** del sistema."
            )
        else:
            st.caption(
                "Las cuentas nuevas entran con rol **visualizador** (solo lectura). "
                "Un administrador puede ampliar los roles desde «Usuarios y roles»."
            )
        with st.form("register_form"):
            full_name = st.text_input("Nombre completo", placeholder="Opcional")
            username = st.text_input("Nombre de usuario")
            email = st.text_input("Email")
            password = st.text_input("Contrasena", type="password", autocomplete="new-password")
            submitted = st.form_submit_button("Crear cuenta")
        if submitted:
            if not (username and email and password):
                st.error("Usuario, email y contrasena son obligatorios.")
            elif len(username.strip()) < 3:
                st.error("El nombre de usuario debe tener al menos 3 caracteres.")
            elif len(password) < 8:
                st.error("La contrasena debe tener al menos 8 caracteres.")
            else:
                ok, msg = _register(email, username, password, full_name)
                (st.success if ok else st.error)(msg)
