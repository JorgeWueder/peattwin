"""Render de tablas, graficos y figuras que EXIGE una interpretacion escrita.

Regla del proyecto: ningun numero se muestra solo. Cada tabla, grafico o figura
va acompanada de una o dos lineas que dicen que significa el resultado.

Aqui esa regla es estructural, no una convencion que se olvida: `interpretacion`
es un argumento *keyword-only* obligatorio. Una tabla sin texto no llega a
ejecutarse.

    from ui import interpret as I

    I.tabla(df, interpretacion="Shapiro rechaza normalidad en las 10 series...")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


def _nota(texto: str) -> None:
    """Bloque de interpretacion, con el mismo aspecto en toda la seccion.

    Se usa un contenedor nativo y no un `<div>` con `unsafe_allow_html`: dentro
    de HTML crudo Streamlit no interpreta Markdown y el `**negrita**` o los
    `codigos` de las interpretaciones saldrian con los asteriscos a la vista.
    """
    with st.container(border=True):
        st.caption(texto)


def nota(texto: str) -> None:
    """Interpretacion suelta, para contenido que no pasa por los helpers de arriba."""
    _nota(texto)


def titulo(texto: str, *, ayuda: str | None = None) -> None:
    st.subheader(texto)
    if ayuda:
        st.caption(ayuda)


def tabla(
    df: pd.DataFrame,
    *,
    interpretacion: str,
    formato: dict[str, str] | None = None,
    altura: int | None = None,
    indice: bool = False,
) -> None:
    """Tabla + su interpretacion. `interpretacion` es obligatoria."""
    if df is None or df.empty:
        st.info("Sin datos para mostrar.")
        _nota(interpretacion)
        return
    datos: Any = df.style.format(formato, na_rep="—") if formato else df
    # `height` solo se pasa si se pidio: Streamlit rechaza height=None.
    extra = {"height": altura} if altura else {}
    st.dataframe(datos, width="stretch", hide_index=not indice, **extra)
    _nota(interpretacion)


def grafico(chart, *, interpretacion: str) -> None:
    """Grafico Altair + su interpretacion."""
    st.altair_chart(chart, width="stretch")
    _nota(interpretacion)


def figura(path: Path, *, titulo_fig: str, interpretacion: str, comando: str = "") -> None:
    """Figura PNG del pipeline. Si falta, explica como regenerarla en vez de fallar."""
    st.markdown(f"**{titulo_fig}**")
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        aviso = f"Falta la figura `{path.name}`."
        if comando:
            aviso += f" Se genera con `{comando}`."
        st.info(aviso)
    _nota(interpretacion)


def metricas(items: Iterable[tuple[str, Any]], *, interpretacion: str) -> None:
    """Fila de `st.metric` + su interpretacion."""
    items = list(items)
    if not items:
        return
    for columna, (etiqueta, valor) in zip(st.columns(len(items)), items):
        columna.metric(etiqueta, valor)
    _nota(interpretacion)


def texto_md(contenido: str | None, *, vacio: str = "Resumen no disponible.") -> None:
    """Render de los RESUMEN_*.md del pipeline dentro de un desplegable."""
    if not contenido:
        st.info(vacio)
        return
    with st.expander("Ver el resumen completo generado por el pipeline"):
        st.markdown(contenido)


def falta_artefacto(nombre: str, comando: str) -> None:
    """Aviso homogeneo cuando un artefacto no esta en disco."""
    st.warning(
        f"No se encuentra **{nombre}**. Esta subseccion lee resultados ya "
        f"calculados; genera el artefacto con:\n\n```bash\n{comando}\n```"
    )
