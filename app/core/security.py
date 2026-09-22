"""Hashing y verificacion de contrasenas con bcrypt.

La aplicacion Streamlit accede a PostgreSQL en el mismo proceso: no hay
frontera HTTP, por lo que no se emiten tokens JWT. La sesion autenticada vive
en `st.session_state` (ver `ui/auth.py`).
"""
from __future__ import annotations

import bcrypt

# bcrypt solo usa los primeros 72 bytes de la contrasena.
_BCRYPT_MAX_BYTES = 72


def _clip(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_clip(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_clip(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
