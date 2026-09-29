"""Render de tablas, graficos y figuras que EXIGE interpretacion y explicabilidad.

Regla del proyecto: ningun numero se muestra solo. Y no basta con una frase: a
cada tabla, grafico o figura le acompanan DOS bloques separados, porque son dos
preguntas distintas y mezclarlas es lo que produce pies de figura que no dicen
nada.

    Interpretacion  -> QUE SIGNIFICA. El hallazgo, la conclusion, la salvedad.
                       Es donde se moja el autor.
    Explicabilidad  -> POR QUE se puede afirmar eso. Que hay en los ejes, en que
                       unidades, sobre que datos, como se calculo, como se lee.
                       Es lo que permite auditar la afirmacion de arriba.

La interpretacion va PRIMERO a proposito: es lo que el lector viene a buscar y
lo que sostiene el argumento del articulo. La explicabilidad va debajo como
respaldo, para quien quiera comprobar de donde sale. Separarlas obliga ademas a
notar cuando falta una de las dos: es facil escribir «el CH4 se predice bien» y
olvidar decir que el eje esta en nmol m-2 s-1.

Aqui la regla es estructural, no una convencion que se olvida: `interpretacion`
y `explicabilidad` son argumentos *keyword-only* obligatorios. Una tabla sin las
dos no llega a ejecutarse.

    from ui import interpret as I

    I.tabla(
        df,
        interpretacion="Ninguna pasa el contraste, de ahi que todo lo que sigue...",
        explicabilidad="Shapiro-Wilk y D'Agostino sobre las 10 series clave...",
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

# Contador para dar una `key` distinta a cada bloque. Streamlit convierte esa
# key en una clase `st-key-<key>` del DOM, y el CSS de `ui/theme.py` engancha
# ahi (`[class*="st-key-ptnota"]`). Antes el estilo se colgaba de «contenedor
# con borde cuyo unico hijo es un caption», que dejo de ser cierto al pasar de
# uno a dos captions: un ancla explicita no se rompe al cambiar el contenido.
_contador = 0


def _key() -> str:
    global _contador
    _contador += 1
    return f"ptnota{_contador}"


def _bloque(interpretacion: str, explicabilidad: str) -> None:
    """Las dos partes, en un contenedor propio y visiblemente separadas.

    Se usa un contenedor nativo y no un `<div>` con `unsafe_allow_html`: dentro
    de HTML crudo Streamlit no interpreta Markdown y el `**negrita**` o los
    `codigos` de los textos saldrian con los asteriscos a la vista.
    """
    with st.container(border=True, key=_key()):
        st.caption(f"**Interpretacion.** {interpretacion}")
        st.caption(f"**Explicabilidad.** {explicabilidad}")


def nota(texto: str) -> None:
    """Anotacion suelta de una sola parte.

    Para comentarios que NO acompanan a una tabla ni a una figura concreta (un
    aviso de trazabilidad, una lectura que cruza dos graficos ya explicados).
    Todo lo que cuelgue de un dato usa los helpers de abajo, que exigen las dos
    partes.
    """
    with st.container(border=True, key=_key()):
        st.caption(texto)


def doble(interpretacion: str, explicabilidad: str) -> None:
    """Las dos partes para contenido que no pasa por los helpers de abajo."""
    _bloque(interpretacion, explicabilidad)


def titulo(texto: str, *, ayuda: str | None = None) -> None:
    st.subheader(texto)
    if ayuda:
        st.caption(ayuda)


def tabla(
    df: pd.DataFrame | Styler,
    *,
    interpretacion: str,
    explicabilidad: str,
    formato: dict[str, str] | None = None,
    altura: int | None = None,
    indice: bool = False,
) -> None:
    """Tabla + sus dos bloques. Ambos son obligatorios.

    Admite tambien un `Styler` ya construido, para las tablas que necesitan
    resaltar filas (`df.style.apply(...)`). Sin esto, esas tablas tendrian que
    llamar a `st.dataframe` por su cuenta y se saldrian de la garantia: el punto
    de este modulo es que no exista una via de escape comoda.
    """
    datos_df = df.data if isinstance(df, Styler) else df
    if datos_df is None or datos_df.empty:
        st.info("Sin datos para mostrar.")
        _bloque(interpretacion, explicabilidad)
        return

    datos: Any = df
    if formato:
        # `Styler.format` es acumulable, asi que se puede encadenar sobre uno
        # que ya traiga `.apply()` sin perder el resaltado.
        datos = df.format(formato, na_rep="—") if isinstance(df, Styler) \
            else df.style.format(formato, na_rep="—")

    # `height` solo se pasa si se pidio: Streamlit rechaza height=None.
    extra = {"height": altura} if altura else {}
    st.dataframe(datos, width="stretch", hide_index=not indice, **extra)
    _bloque(interpretacion, explicabilidad)


def grafico(chart, *, interpretacion: str, explicabilidad: str) -> None:
    """Grafico Altair + sus dos bloques."""
    st.altair_chart(chart, width="stretch")
    _bloque(interpretacion, explicabilidad)


def figura(
    path: Path,
    *,
    titulo_fig: str,
    interpretacion: str,
    explicabilidad: str,
    comando: str = "",
) -> None:
    """Figura PNG del pipeline. Si falta, explica como regenerarla en vez de fallar.

    Los dos bloques se muestran aunque la imagen no este: describen un resultado
    que existe en el CSV correspondiente, y ocultarlos dejaria al lector sin
    saber siquiera que deberia haber ahi.
    """
    st.markdown(f"**{titulo_fig}**")
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        aviso = f"Falta la figura `{path.name}`."
        if comando:
            aviso += f" Se genera con `{comando}`."
        st.info(aviso)
    _bloque(interpretacion, explicabilidad)


def metricas(
    items: Iterable[tuple[str, Any]], *, interpretacion: str, explicabilidad: str
) -> None:
    """Fila de `st.metric` + sus dos bloques."""
    items = list(items)
    if not items:
        return
    for columna, (etiqueta, valor) in zip(st.columns(len(items)), items):
        columna.metric(etiqueta, valor)
    _bloque(interpretacion, explicabilidad)


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
