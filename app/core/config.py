"""Configuracion central de la aplicacion.

Los valores se leen de variables de entorno y, si existe, del archivo `.env`
ubicado en la raiz del repositorio (desde donde se ejecuta Streamlit).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Ruta absoluta al .env de la raiz: asi funciona aunque el proceso arranque
# desde otro directorio de trabajo.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "PeatTwin"
    ENVIRONMENT: str = "development"

    # --- PostgreSQL local ---
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "peattwin"

    # --- ML / prediccion ---
    # Ruta explicita al modelo desplegado; si se deja vacia se resuelve desde
    # ml_models.file_path (BD) y, si no, desde <repo>/ml/models/modelo_final.{pkl,h5}.
    MODEL_FILE: str = ""

    @property
    def project_root(self) -> Path:
        # app/core/config.py -> parents[2] == raiz del repo
        return Path(__file__).resolve().parents[2]

    @property
    def models_dir(self) -> Path:
        return self.project_root / "ml" / "models"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def database_url(self) -> str:
        """URL de conexion SQLAlchemy hacia la base de datos PostgreSQL local."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache
def get_settings() -> "Settings":
    return Settings()


settings = get_settings()
