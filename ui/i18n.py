"""Selector de idioma de la interfaz. La logica de traduccion vive en `app.core.i18n`.

El idioma va en `st.session_state`. `auth.logout()` barre todo el estado, asi que
`keep_on_logout()` / `restore()` lo preservan: cerrar sesion no cambia el idioma.
"""
from __future__ import annotations

import streamlit as st

from app.core.i18n import LANGS, SESSION_KEY, force, is_en, lang, tr

__all__ = ["LANGS", "force", "is_en", "lang", "tr", "keep_on_logout", "restore", "switch"]

_LABELS = {"es": "Español", "en": "English"}


def keep_on_logout() -> str:
    return lang()


def restore(value: str) -> None:
    st.session_state[SESSION_KEY] = value


def switch(*, key: str) -> None:
    """Selector de idioma. `key` distinto por ubicacion (login / barra lateral)."""
    current = lang()
    choice = st.segmented_control(
        tr("Idioma", "Language"),
        LANGS,
        default=current,
        format_func=lambda code: _LABELS[code],
        key=key,
        label_visibility="collapsed",
    )
    if choice and choice != current:
        st.session_state[SESSION_KEY] = choice
        st.rerun()
