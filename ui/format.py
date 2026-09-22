"""Formateo de numeros y fechas para la interfaz.

Mantiene el criterio del dashboard anterior: guion largo para los nulos y un
numero fijo de decimales por metrica.
"""
from __future__ import annotations

from datetime import date, datetime

DASH = "—"


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return DASH
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return DASH
    return f"{v:.{digits}f}"


def fmt_signed(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return DASH
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return DASH
    return f"{'+' if v > 0 else ''}{v:.{digits}f}"


def fmt_seconds(value: float | int | None) -> str:
    if value is None:
        return DASH
    v = float(value)
    if v < 60:
        return f"{v:.1f} s"
    minutes, seconds = divmod(v, 60)
    return f"{int(minutes)} min {int(seconds)} s"


def fmt_pct(value: float | None, digits: int = 0) -> str:
    return DASH if value is None else f"{value * 100:.{digits}f} %"


def fmt_date(value: date | datetime | str | None) -> str:
    if value is None:
        return DASH
    if isinstance(value, str):
        return value[:10]
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return value.isoformat()
