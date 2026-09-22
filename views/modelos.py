"""Comparacion de los 5 modelos (Pasos 5-8) y descarga de informes.

La fusion de las filas de regresion y clasificacion por nombre base ya estaba
resuelta en `app.reports.common.model_rows`, que ademas trae la matriz de
confusion del modelo desplegado: se reutiliza en lugar de replicarla aqui.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import pandas as pd
import streamlit as st

from app.core.errors import DomainError
from app.reports import common as C
from app.reports import docx_report, pdf_report, xlsx_report
from app.services import site_service
from ui.db import session_scope

FIGURES = [
    (
        "matrices_confusion.png",
        "Matrices de confusion (holdout 2018)",
        "Clase positiva = sumidero (minoritaria, ~32 %). Debajo de cada matriz se "
        "muestran F1 y recall de sumidero.",
    ),
    (
        "curvas_roc_comparacion.png",
        "Curvas ROC — sumidero vs fuente",
        "AUC entre 0.85 (CNN-LSTM) y 0.92 (Stacking). Las 5 curvas sobre el mismo eje.",
    ),
    (
        "heatmap_comparativo_metricas.png",
        "Heatmap comparativo de metricas",
        "Verde = mejor, rojo = peor. Cada columna se normaliza min-max entre los 5 modelos.",
    ),
]

_CRITERIO = (
    "Criterio de seleccion: `combined = 0.5·reg_score_norm + 0.5·(0.5·F1_macro + 0.5·AUC)` "
    "— 50 % regresion (R² medio de CO₂ y CH₄, normalizado min-max entre modelos) y 50 % "
    "clasificacion (F1_macro y AUC, no accuracy, por el desbalance de clases)."
)

_COLUMNS = {
    "name": "Modelo",
    "familia": "Familia",
    "version": "Version",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "nse": "NSE",
    "kge": "KGE",
    "accuracy": "Accuracy",
    "f1": "F1",
    "auc": "AUC",
    "train_time_s": "t entren. (s)",
}


@st.cache_data(ttl=300, show_spinner=False)
def _comparison() -> list[dict]:
    with session_scope() as db:
        rows = [asdict(r) for r in C.model_rows(db)]
    for r in rows:
        r["familia"] = "hibrido" if r["algorithm_type"] == "HYBRID" else "clasico"
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def _sites() -> list[dict]:
    with session_scope() as db:
        rows = [
            {"id": s.id, "name": s.name, "fluxnet_id": s.fluxnet_id}
            for s in site_service.list_sites(db)
        ]
    rows.sort(key=lambda s: (s["fluxnet_id"] != "DE-Zrk", s["id"]))
    return rows


def _slug(site: dict) -> str:
    return (site["fluxnet_id"] or site["name"]).replace(" ", "_").replace("/", "-")


def _highlight_active(row: pd.Series, active_names: set[str]) -> list[str]:
    style = "background-color: rgba(5, 150, 105, 0.12); font-weight: 600;"
    return [style if row["Modelo"] in active_names else "" for _ in row]


_PDF_MIME = "application/pdf"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@st.cache_data(ttl=600, show_spinner=False)
def build_report(kind: str, site_id: int, **kwargs) -> bytes:
    """Genera un informe bajo demanda. Los tres builders devuelven `bytes`."""
    with session_scope() as db:
        if kind == "pdf":
            return pdf_report.build(db, site_id=site_id, start=None, end=None)
        if kind == "docx":
            return docx_report.build(db, site_id=site_id, **kwargs)
        return xlsx_report.build(db, site_id=site_id)


def _report_button(kind: str, label: str, filename: str, mime: str, site_id: int, **kwargs) -> None:
    """Dos pasos: generar (caro) y luego descargar, para no penalizar cada rerun."""
    state_key = f"report_{kind}_{site_id}"
    if st.button(f"Generar {label}", key=f"gen_{state_key}", width="stretch"):
        try:
            with st.spinner(f"Generando {label}…"):
                st.session_state[state_key] = build_report(kind, site_id, **kwargs)
        except DomainError as exc:
            st.session_state.pop(state_key, None)
            st.error(str(exc))
    content = st.session_state.get(state_key)
    if content:
        st.download_button(
            f"Descargar {label}",
            content,
            file_name=filename,
            mime=mime,
            key=f"dl_{state_key}",
            type="primary",
            width="stretch",
        )


def _reports_section(site: dict) -> None:
    st.subheader("Informes descargables")
    st.caption(
        "Se generan al vuelo desde la base de datos y los artefactos de `ml/`. "
        "Cada tabla y figura lleva su parrafo de interpretacion. La primera "
        "generacion tarda unos segundos porque ejecuta el modelo sobre la serie."
    )
    slug = _slug(site)
    col_pdf, col_docx, col_xlsx = st.columns(3)
    with col_pdf:
        _report_button("pdf", "resumen ejecutivo (.pdf)", f"peattwin_resumen_{slug}.pdf",
                       _PDF_MIME, site["id"])
    with col_docx:
        _report_button("docx", "informe tecnico (.docx)",
                       f"peattwin_informe_tecnico_{slug}.docx", _DOCX_MIME, site["id"], folds=5)
    with col_xlsx:
        _report_button("xlsx", "datos y metricas (.xlsx)", f"peattwin_datos_{slug}.xlsx",
                       _XLSX_MIME, site["id"])


def render() -> None:
    st.title("Comparacion de modelos (Pasos 5–8)")
    st.caption(
        "Metricas de regresion (media de CO₂ y CH₄) y de clasificacion en el holdout "
        "de 2018. La fila resaltada es el modelo desplegado."
    )

    sites = _sites()
    if sites:
        if len(sites) > 1:
            site = st.selectbox(
                "Sitio de los informes",
                sites,
                format_func=lambda s: f"{s['name']} · {s['fluxnet_id']}" if s["fluxnet_id"] else s["name"],
            )
        else:
            site = sites[0]
        _reports_section(site)
        st.divider()

    rows = _comparison()
    if not rows:
        st.warning(
            "No hay modelos registrados. Ejecuta `python -m ml.training.run_training` "
            "para entrenar los 5 modelos y poblar `ml_models` / `ml_model_metrics`."
        )
        return

    active_names = {r["name"] for r in rows if r["is_active"]}
    df = pd.DataFrame(rows)[list(_COLUMNS)].rename(columns=_COLUMNS)

    st.subheader("Tabla comparativa")
    st.dataframe(
        df.style.apply(_highlight_active, axis=1, active_names=active_names).format(
            {
                "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}",
                "NSE": "{:.3f}", "KGE": "{:.3f}", "Accuracy": "{:.3f}",
                "F1": "{:.3f}", "AUC": "{:.3f}", "t entren. (s)": "{:.1f}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(_CRITERIO)
    st.caption(
        "RMSE/MAE/R²/NSE/KGE son la media de los objetivos CO₂ y CH₄ (el detalle por "
        "objetivo esta en `ml/training/resultados_comparativa.csv`). El KGE del NEE no "
        "esta definido (media ≈ 0)."
    )

    st.subheader("Figuras del estudio (Pasos 5–6)")
    cols = st.columns(2)
    for i, (filename, title, caption) in enumerate(FIGURES):
        path = C.TRAIN_FIG / filename
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                st.info(
                    f"Falta `{path.relative_to(C.settings.project_root)}`. La genera "
                    "`python -m ml.training.run_training`."
                )
            st.caption(caption)
