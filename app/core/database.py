"""Motor y fabrica de sesiones de SQLAlchemy.

El `engine` se crea una sola vez por proceso. Las vistas de Streamlit abren y
cierran una sesion por rerun con el contextmanager `ui.db.session_scope()`.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)
