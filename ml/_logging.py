"""Captura a fichero de la salida del pipeline ML.

Los scripts de `ml/` comunican su avance con `print()`. Esta utilidad redirige
esa salida a `ml/logs/<etapa>_<timestamp>.log` **sin dejar de escribirla en
consola**, de modo que cada script solo tiene que llamar a `setup()` al empezar
su `main()` y no cambia ni una linea mas.

    from ml import _logging

    def main() -> None:
        _logging.setup("entrenamiento")
        ...

Los ficheros resultantes son los que lee la subseccion «Logs» de Motor IA.
"""
from __future__ import annotations

import atexit
import io
import sys
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"

_activo: Path | None = None


class _Tee(io.TextIOBase):
    """Escribe en la consola y en el fichero a la vez."""

    def __init__(self, consola, fichero) -> None:
        self._consola = consola
        self._fichero = fichero

    def write(self, texto: str) -> int:
        self._consola.write(texto)
        self._fichero.write(texto)
        self._fichero.flush()
        return len(texto)

    def flush(self) -> None:
        self._consola.flush()
        self._fichero.flush()


def setup(etapa: str) -> Path:
    """Empieza a duplicar stdout/stderr en `ml/logs/<etapa>_<timestamp>.log`.

    Devuelve la ruta del fichero. Idempotente por proceso: si ya hay una captura
    activa devuelve la que estaba, para que llamar a dos etapas seguidas en el
    mismo proceso (el boton de reentrenar) no anide redirecciones.
    """
    global _activo
    if _activo is not None:
        return _activo

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    destino = LOGS_DIR / f"{etapa}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fichero = destino.open("w", encoding="utf-8")

    cabecera = f"=== {etapa} · {datetime.now():%Y-%m-%d %H:%M:%S} ===\n"
    fichero.write(cabecera)

    consola_out, consola_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(consola_out, fichero)
    sys.stderr = _Tee(consola_err, fichero)

    def _cerrar() -> None:
        global _activo
        sys.stdout, sys.stderr = consola_out, consola_err
        if not fichero.closed:
            fichero.write(f"=== fin · {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            fichero.close()
        _activo = None

    atexit.register(_cerrar)
    setup.cerrar = _cerrar  # type: ignore[attr-defined]
    _activo = destino
    return destino


def cerrar() -> None:
    """Cierra la captura activa (lo usa el reentrenamiento en proceso)."""
    fn = getattr(setup, "cerrar", None)
    if fn is not None:
        fn()


def ficheros() -> list[Path]:
    """Logs existentes, del mas reciente al mas antiguo."""
    if not LOGS_DIR.exists():
        return []
    return sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
