"""Pantalla de acceso: entrar y crear cuenta.

El registro replica la regla que tenia la API: el PRIMER usuario del sistema
recibe el rol `admin` y `is_superuser`; el resto entran como `visualizador`.
"""
from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import IntegrityError

from app.services import auth_service
from ui import auth, i18n, theme
from ui.db import session_scope
from ui.i18n import tr


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
            return False, tr(
                "El email o el nombre de usuario ya estan registrados.",
                "The email or username is already registered.",
            )
    rol = "admin" if first_user else "visualizador"
    return True, tr(
        f"Cuenta creada con rol «{rol}». Ya puedes entrar.",
        f"Account created with role «{rol}». You can sign in now.",
    )


def render() -> None:
    # Solo estilos: el ancho de dashboard deja este formulario estirado.
    theme.inject_login()

    st.markdown("## PeatTwin")
    i18n.switch(key="lang_login")
    st.caption(tr(
        "Gemelo Digital de Turberas — restauracion hidrologica y balance de "
        "carbono CO₂–CH₄ del sitio DE-Zrk (Zarnekow).",
        "Peatland Digital Twin — hydrological restoration and CO₂–CH₄ carbon "
        "balance of the DE-Zrk (Zarnekow) site.",
    ))

    entrar, crear = st.tabs([tr("Entrar", "Sign in"), tr("Crear una cuenta", "Create an account")])

    with entrar:
        with st.form("login_form"):
            identifier = st.text_input(tr("Usuario o email", "Username or email"), autocomplete="username")
            password = st.text_input(tr("Contrasena", "Password"), type="password", autocomplete="current-password")
            submitted = st.form_submit_button(tr("Entrar", "Sign in"), type="primary")
        if submitted:
            if not identifier or not password:
                st.error(tr("Introduce usuario y contrasena.", "Enter your username and password."))
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
            st.info(tr(
                "No hay ningun usuario todavia: esta primera cuenta sera el "
                "**administrador** del sistema.",
                "There are no users yet: this first account will be the "
                "system **administrator**.",
            ))
        else:
            st.caption(tr(
                "Las cuentas nuevas entran con rol **visualizador** (solo lectura). "
                "Un administrador puede ampliar los roles desde «Usuarios y roles».",
                "New accounts get the **visualizador** (read-only) role. "
                "An administrator can extend roles from «Users and roles».",
            ))
        with st.form("register_form"):
            full_name = st.text_input(tr("Nombre completo", "Full name"), placeholder=tr("Opcional", "Optional"))
            username = st.text_input(tr("Nombre de usuario", "Username"))
            email = st.text_input("Email")
            password = st.text_input(tr("Contrasena", "Password"), type="password", autocomplete="new-password")
            submitted = st.form_submit_button(tr("Crear cuenta", "Create account"))
        if submitted:
            if not (username and email and password):
                st.error(tr("Usuario, email y contrasena son obligatorios.", "Username, email and password are required."))
            elif len(username.strip()) < 3:
                st.error(tr("El nombre de usuario debe tener al menos 3 caracteres.", "The username must be at least 3 characters long."))
            elif len(password) < 8:
                st.error(tr("La contrasena debe tener al menos 8 caracteres.", "The password must be at least 8 characters long."))
            else:
                ok, msg = _register(email, username, password, full_name)
                (st.success if ok else st.error)(msg)
