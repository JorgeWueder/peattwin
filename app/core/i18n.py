"""Idioma activo (es / en) y traduccion en linea `tr("español", "english")`.

Los dos idiomas conviven junto al codigo que los usa: un cambio de redaccion no
puede dejar una traduccion huerfana en un catalogo aparte, y sirve tambien para
f-strings, que un catalogo clave -> texto no puede cubrir.

Resolucion del idioma, por orden:
  1. `force(lang)`: lo usan los generadores de informes (PDF, Word, Excel), que
     se ejecutan dentro de funciones cacheadas y reciben el idioma como argumento.
  2. `st.session_state["lang"]`: el selector de la interfaz.
  3. `DEFAULT`.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

LANGS = ("es", "en")
DEFAULT = "es"
SESSION_KEY = "lang"

_override: ContextVar[str | None] = ContextVar("peattwin_lang_override", default=None)


def normalize(value: str | None) -> str:
    return value if value in LANGS else DEFAULT


def lang() -> str:
    forced = _override.get()
    if forced:
        return normalize(forced)
    try:
        import streamlit as st

        return normalize(st.session_state.get(SESSION_KEY, DEFAULT))
    except Exception:
        return DEFAULT


def is_en() -> bool:
    return lang() == "en"


def tr(es: str, en: str) -> str:
    return en if lang() == "en" else es


@contextmanager
def force(value: str):
    token = _override.set(normalize(value))
    try:
        yield
    finally:
        _override.reset(token)
