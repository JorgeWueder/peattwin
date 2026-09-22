"""Graficos Altair de las vistas.

En tema claro la paleta es la misma que usan las figuras de los informes
(`app/reports/figures.py`), para que la interfaz y el PDF/Word se vean iguales.
En tema oscuro los neutros se invierten (ver `_paleta`): el acento y el papel
del color se conservan, pero la tinta de «observado» no puede seguir siendo
casi negra o la serie desaparece del grafico.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

EMERALD = "#059669"   # predicho
INK = "#18181b"       # observado
ORANGE = "#ea580c"    # CH4 en la curva de respuesta
GRID = "#e4e4e7"

_HEIGHT = 260

# Tipografia de los ejes: la misma monoespaciada tabular que usa la interfaz
# para cualquier cifra (ver `ui/theme.py`). Las etiquetas numericas de un eje
# son datos, y en mono no cambian de ancho al cambiar de valor.
_MONO = "Geist Mono, ui-monospace, Menlo, monospace"


# ===================================================== paleta segun el tema
# Las CUATRO constantes de arriba no se tocan: `app/reports/figures.py` las
# importa para las figuras del PDF y del Word, que siempre van sobre papel
# blanco. Lo que cambia con el tema es solo lo que resuelve la interfaz.
#
# El motivo es un fallo real: la serie «observado» iba en tinta #18181b y
# sobre el fondo oscuro (#09090b) desaparecia por completo, dejando el grafico
# con una sola curva visible de las dos que compara. Una serie que no se ve es
# un dato perdido, no una preferencia estetica.

_CLARO = {
    "observado": INK,
    "predicho": EMERALD,
    "ch4": ORANGE,
    "grid": GRID,
    "label": "#71717a",
    "legend": "#3f3f46",
    "marcador": "#71717a",
    "cero": "#a1a1aa",
    "texto": INK,
}

_OSCURO = {
    "observado": "#e4e4e7",   # zinc-200: el neutro se invierte, no se pierde
    "predicho": "#34d399",    # esmeralda mas luminosa para el mismo contraste
    "ch4": "#fb923c",
    "grid": "#27272a",
    "label": "#a1a1aa",
    "legend": "#d4d4d8",
    "marcador": "#71717a",
    "cero": "#52525b",
    "texto": "#fafafa",
}


def _paleta() -> dict[str, str]:
    """Paleta del tema activo. Si no se puede leer el tema, asume claro."""
    try:
        oscuro = st.context.theme.type == "dark"
    except Exception:
        oscuro = False
    return _OSCURO if oscuro else _CLARO


def _base(df: pd.DataFrame) -> alt.Chart:
    return alt.Chart(df).properties(height=_HEIGHT)


def _axis(**kwargs) -> alt.Axis:
    """Eje del sistema: rejilla tenue, sin linea de dominio, etiquetas en mono.

    El recuadro completo que Altair dibuja por defecto compite con el marco de
    1px que el contenedor ya aporta.
    """
    p = _paleta()
    opciones = dict(
        gridColor=p["grid"],
        gridOpacity=0.7,
        domain=False,
        tickSize=0,
        labelFont=_MONO,
        labelFontSize=10,
        labelColor=p["label"],
        labelPadding=6,
        titleFontSize=11,
        titleColor=p["label"],
        titleFontWeight=500,
        titlePadding=10,
    )
    opciones.update(kwargs)
    return alt.Axis(**opciones)


def _legend() -> alt.Legend:
    """Leyenda por encima del area de trazado y alineada con el titulo.

    Antes iba en `top-right`, que la dibuja DENTRO del grafico: en esta serie
    se superponia a los picos de 2018.
    """
    return alt.Legend(
        title=None,
        orient="top",
        direction="horizontal",
        legendX=0,
        legendY=-24,
        labelFontSize=11,
        labelColor=_paleta()["legend"],
        symbolStrokeWidth=2.5,
        symbolSize=90,
    )


def series_chart(
    df: pd.DataFrame, *, unit: str, zero_line: bool = False
) -> alt.LayerChart:
    """Serie observado vs predicho. `df` con columnas date / observado / predicho."""
    long = df.melt(
        id_vars="date",
        value_vars=["observado", "predicho"],
        var_name="serie",
        value_name="valor",
    )
    p = _paleta()
    line = (
        _base(long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None, axis=_axis(format="%b %Y", grid=False)),
            y=alt.Y("valor:Q", title=unit, axis=_axis()),
            color=alt.Color(
                "serie:N",
                title=None,
                scale=alt.Scale(
                    domain=["observado", "predicho"],
                    range=[p["observado"], p["predicho"]],
                ),
                legend=_legend(),
            ),
            strokeWidth=alt.condition(
                alt.datum.serie == "predicho", alt.value(2.0), alt.value(1.0)
            ),
            tooltip=[
                alt.Tooltip("date:T", title="fecha"),
                alt.Tooltip("serie:N", title="serie"),
                alt.Tooltip("valor:Q", title="valor", format=".3f"),
            ],
        )
    )
    layers = [line]
    if zero_line:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=p["cero"], strokeDash=[4, 3])
            .encode(y="y:Q")
        )
    # Sin `.interactive()`: la rueda del raton debe desplazar la pagina, no
    # hacer zoom en el grafico.
    return alt.layer(*layers)


def response_curve_chart(
    df: pd.DataFrame, *, metric: str, marker: float, unit: str
) -> alt.LayerChart:
    """Curva de respuesta al nivel freatico. `df` con columnas wtd / co2 / ch4."""
    p = _paleta()
    color = p["predicho"] if metric == "co2" else p["ch4"]
    label = "CO₂ · NEE" if metric == "co2" else "CH₄ · FCH4"
    line = (
        _base(df)
        .mark_line(point=True, color=color, strokeWidth=2)
        .encode(
            x=alt.X("wtd:Q", title="Nivel freatico objetivo (cm)", axis=_axis()),
            y=alt.Y(f"{metric}:Q", title=f"{label} ({unit})", axis=_axis()),
            tooltip=[
                alt.Tooltip("wtd:Q", title="WTD (cm)", format=".0f"),
                alt.Tooltip(f"{metric}:Q", title=label, format=".3f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"wtd": [marker]}))
        .mark_rule(color=p["marcador"], strokeDash=[4, 3])
        .encode(x="wtd:Q")
    )
    layers = [line, rule]
    if metric == "co2":
        layers.append(
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=p["cero"], strokeDash=[2, 2])
            .encode(y="y:Q")
        )
    return alt.layer(*layers)


# ===================================================== helpers de la seccion Motor IA
# Sustituyen a `df.style.background_gradient()`, que exigiria matplotlib como
# dependencia extra: todo se resuelve con Altair, que ya usa el resto de la app.

def heatmap(
    df: pd.DataFrame,
    *,
    dominio: tuple[float, float] = (-1.0, 1.0),
    esquema: str = "redyellowgreen",
    decimales: str = ".2f",
    titulo_valor: str = "valor",
    altura: int = 460,
) -> alt.LayerChart:
    """Matriz cuadrada (correlacion, metricas normalizadas) como heatmap.

    `df` viene con las variables en el indice y en las columnas.
    """
    largo = df.reset_index().melt(
        id_vars=df.index.name or "index", var_name="columna", value_name="valor"
    )
    largo = largo.rename(columns={df.index.name or "index": "fila"})
    orden = list(df.index)

    celdas = (
        alt.Chart(largo)
        .mark_rect()
        .encode(
            x=alt.X("columna:N", title=None, sort=orden,
                    axis=_axis(labelAngle=-45, labelLimit=140, grid=False)),
            y=alt.Y("fila:N", title=None, sort=orden, axis=_axis(labelLimit=140, grid=False)),
            color=alt.Color(
                "valor:Q",
                title=titulo_valor,
                scale=alt.Scale(scheme=esquema, domain=list(dominio)),
            ),
            tooltip=[
                alt.Tooltip("fila:N", title="fila"),
                alt.Tooltip("columna:N", title="columna"),
                alt.Tooltip("valor:Q", title=titulo_valor, format=decimales),
            ],
        )
    )
    # El texto de las celdas va sobre el color del heatmap, que no cambia con
    # el tema (escala propia), asi que se queda en tinta en ambos modos.
    etiquetas = celdas.mark_text(fontSize=9).encode(
        text=alt.Text("valor:Q", format=decimales),
        color=alt.value(INK),
    )
    return alt.layer(celdas, etiquetas).properties(height=altura)


def barras_agrupadas(
    df: pd.DataFrame, *, x: str, y: str, color: str, titulo_y: str, altura: int = 300
) -> alt.Chart:
    """Comparacion de una metrica entre modelos, agrupada por serie."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=_axis(labelAngle=-30, labelLimit=160, grid=False)),
            y=alt.Y(f"{y}:Q", title=titulo_y, axis=_axis()),
            # Leyenda lateral por defecto, no la superior de `_legend()`: en un
            # grafico de barras no hay solape que evitar, y la superior con
            # desplazamiento negativo se recortaria a esta altura.
            color=alt.Color(
                f"{color}:N", title=None,
                scale=alt.Scale(range=[_paleta()["predicho"], _paleta()["ch4"],
                                       _paleta()["observado"]]),
            ),
            xOffset=f"{color}:N",
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{color}:N"),
                     alt.Tooltip(f"{y}:Q", format=".3f")],
        )
        .properties(height=altura)
    )


def boxplot_folds(
    df: pd.DataFrame, *, modelo: str, valor: str, titulo_y: str, altura: int = 300
) -> alt.Chart:
    """Dispersion de una metrica entre folds, un box por modelo."""
    return (
        alt.Chart(df)
        .mark_boxplot(extent="min-max", color=_paleta()["predicho"], size=34)
        .encode(
            x=alt.X(f"{modelo}:N", title=None, axis=_axis(labelAngle=-30, labelLimit=160, grid=False)),
            y=alt.Y(f"{valor}:Q", title=titulo_y, axis=_axis()),
            tooltip=[alt.Tooltip(f"{modelo}:N"), alt.Tooltip(f"{valor}:Q", format=".3f")],
        )
        .properties(height=altura)
    )


def barras_horizontales(
    df: pd.DataFrame, *, etiqueta: str, valor: str, titulo_x: str,
    color: str | None = None, altura: int = 340,
) -> alt.Chart:
    """Importancia de variables o pesos del meta-modelo, ordenados de mayor a menor.

    `color=None` toma el acento del tema activo. Se admite un color explicito
    por compatibilidad con las llamadas que ya lo pasaban.
    """
    return (
        alt.Chart(df)
        .mark_bar(color=color or _paleta()["predicho"])
        .encode(
            x=alt.X(f"{valor}:Q", title=titulo_x, axis=_axis()),
            y=alt.Y(f"{etiqueta}:N", title=None, sort="-x", axis=_axis(labelLimit=220, grid=False)),
            tooltip=[alt.Tooltip(f"{etiqueta}:N"), alt.Tooltip(f"{valor}:Q", format=".4f")],
        )
        .properties(height=altura)
    )
