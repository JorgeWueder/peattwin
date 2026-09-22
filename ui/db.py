"""Acceso a la base de datos desde las vistas de Streamlit.

Cada rerun del script abre y cierra su propia sesion. El `engine` de
`app.core.database` se crea una unica vez por proceso y trae `pool_pre_ping`,
asi que NO se envuelve en `st.cache_resource`.

    with session_scope() as db:
        sitios = [{"id": s.id, "name": s.name} for s in site_service.list_sites(db)]

Regla importante: los servicios devuelven entidades SQLAlchemy. Fuera del
bloque `with`, cualquier acceso perezoso (`user.roles`, `role.permissions`)
lanza `DetachedInstanceError`. Convierte a `dict` / `DataFrame` DENTRO del
bloque y no guardes nunca un objeto ORM en `st.session_state`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from app.core.database import SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Sesion transaccional: commit al salir, rollback si hay excepcion."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
