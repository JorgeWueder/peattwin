"""Errores de dominio, independientes de cualquier framework web.

Sustituyen a las `HTTPException` que usaban los servicios cuando la aplicacion
exponia una API FastAPI. Las vistas de Streamlit los capturan y deciden como
presentarlos (`st.error`, `st.warning` o un aviso explicativo).
"""
from __future__ import annotations

from typing import Literal

Kind = Literal["not_found", "unavailable", "invalid"]


class DomainError(Exception):
    """Fallo esperable de la capa de dominio, con un motivo clasificado."""

    def __init__(self, message: str, kind: Kind = "invalid") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind

    def __str__(self) -> str:
        return self.message
