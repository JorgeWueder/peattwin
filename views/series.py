"""Series temporales: CO₂ (NEE) y CH₄ (FCH4) observados frente al modelo activo.

Backtest sobre las variables REALES de cada dia (no la climatologia del
simulador), tal y como lo hacia la vista equivalente del dashboard anterior.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.core.errors import DomainError
from app.reports.common import regression_scores
from app.services import model_service, site_service
from ui import format as F
from ui.charts import series_chart
from ui.db import session_scope

DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2018, 12, 31)


@st.cache_data(ttl=300, show_spinner=False)
def _sites() -> list[dict]:
    with session_scope() as db:
        rows = [
            {"id": s.id, "name": s.name, "fluxnet_id": s.fluxnet_id}
            for s in site_service.list_sites(db)
        ]
    # DE-Zrk primero: es el unico con dataset procesado.
    rows.sort(key=lambda s: (s["fluxnet_id"] != "DE-Zrk", s["id"]))
    return rows


@st.cache_data(ttl=300, show_spinner="Ejecutando el modelo activo sobre la serie…")
def _series(site_id: int, start: date, end: date) -> dict:
    with session_scope() as db:
        resp = model_service.predicted_series(db, site_id=site_id, start=start, end=end)
    return {
        "model_name": resp.model_name,
        "model_version": resp.model_version,
        "points": [p.model_dump() for p in resp.points],
    }


def render() -> None:
    st.title("Series temporales — observado vs predicho")
    st.caption(
        "CO₂ (NEE) y CH₄ (FCH4) medidos frente a la prediccion del modelo activo, "
        "usando las variables reales de cada dia (backtest)."
    )

    sites = _sites()
    if not sites:
        st.warning(
            "No hay sitios en la base de datos. Ejecuta `python -m ml.etl.load_to_db` "
            "para cargar DE-Zrk y sus observaciones."
        )
        return

    col_site, col_start, col_end = st.columns([2, 1, 1])
    with col_site:
        site = st.selectbox(
            "Sitio",
            sites,
            format_func=lambda s: f"{s['name']} · {s['fluxnet_id']}" if s["fluxnet_id"] else s["name"],
        )
    with col_start:
        start = st.date_input("Desde", DEFAULT_START, format="YYYY-MM-DD")
    with col_end:
        end = st.date_input("Hasta", DEFAULT_END, format="YYYY-MM-DD")

    if start > end:
        st.error("La fecha inicial es posterior a la final.")
        return

    try:
        data = _series(site["id"], start, end)
    except DomainError as exc:
        if exc.kind == "not_found":
            st.info(
                f"**Sin serie predicha para {site['name']}.** Solo el sitio "
                "**DE-Zrk (Zarnekow)** tiene un dataset procesado con el que ejecutar "
                "el modelo. Selecciona ese sitio para ver la comparacion observado vs "
                "predicho."
            )
        else:
            st.error(str(exc))
        return

    points = data["points"]
    if not points:
        st.info("Sin datos en el rango seleccionado.")
        return

    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"])

    co2_rmse, co2_r2 = regression_scores(df["nee_observed"].tolist(), df["nee_predicted"].tolist())
    ch4_rmse, ch4_r2 = regression_scores(df["fch4_observed"].tolist(), df["fch4_predicted"].tolist())
    acc = float((df["class_observed"] == df["class_predicted"]).mean())

    cols = st.columns(6)
    cols[0].metric("Dias", len(df))
    cols[1].metric("RMSE CO₂", F.fmt(co2_rmse, 2))
    cols[2].metric("R² CO₂", F.fmt(co2_r2, 2))
    cols[3].metric("RMSE CH₄", F.fmt(ch4_rmse, 1))
    cols[4].metric("R² CH₄", F.fmt(ch4_r2, 2))
    cols[5].metric("Clase acierto", F.fmt_pct(acc))

    st.subheader("CO₂ — NEE (gC m⁻² d⁻¹)")
    st.caption(
        "Negativo = sumidero. El modelo suaviza los picos diarios "
        "(R² de NEE ≈ 0.24 en el holdout de 2018)."
    )
    st.altair_chart(
        series_chart(
            df.rename(columns={"nee_observed": "observado", "nee_predicted": "predicho"})[
                ["date", "observado", "predicho"]
            ],
            unit="gC m⁻² d⁻¹",
            zero_line=True,
        ),
        width="stretch",
    )

    st.subheader("CH₄ — FCH4 (nmol CH₄ m⁻² s⁻¹)")
    st.caption("Flujo siempre positivo; el modelo lo sigue bien (R² de FCH4 ≈ 0.84).")
    st.altair_chart(
        series_chart(
            df.rename(columns={"fch4_observed": "observado", "fch4_predicted": "predicho"})[
                ["date", "observado", "predicho"]
            ],
            unit="nmol m⁻² s⁻¹",
        ),
        width="stretch",
    )

    st.caption(f"Modelo activo: **{data['model_name']} · {data['model_version']}**")
