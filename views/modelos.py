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
from ui import interpret as I
from ui.db import session_scope

# ------------------------------------------------------------------- lecturas
# Las interpretaciones se calculan sobre las filas que vienen de la base de
# datos, no se escriben a mano: un reentrenamiento cambia las cifras de la tabla
# y un pie de figura fijo ("AUC entre 0.85 y 0.92") pasaria a contradecirla.

_DASH = "—"


def _fnum(v, d=3) -> str:
    return _DASH if v is None else f"{float(v):.{d}f}"


def _mejor(rows: list[dict], campo: str, mayor_mejor: bool = True) -> dict | None:
    validos = [r for r in rows if r.get(campo) is not None]
    if not validos:
        return None
    return (max if mayor_mejor else min)(validos, key=lambda r: float(r[campo]))


def _activo(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r["is_active"]), None)


def _lectura_tabla(rows: list[dict], active_names: set[str]) -> str:
    act = _activo(rows)
    mejor_r2 = _mejor(rows, "r2")
    mejor_auc = _mejor(rows, "auc")
    mas_rapido = _mejor(rows, "train_time_s", mayor_mejor=False)
    quien = ", ".join(sorted(active_names)) or "ninguno"
    coincide = (
        act is not None and mejor_r2 is not None and mejor_auc is not None
        and act["name"] == mejor_r2["name"] == mejor_auc["name"]
    )
    if coincide:
        matiz = (
            "El modelo desplegado gana a la vez en regresion y en clasificacion, asi que el "
            "criterio combinado no arbitra nada: la eleccion habria sido la misma con "
            "cualquiera de los dos pesos."
        )
    else:
        n_r2 = mejor_r2["name"] if mejor_r2 else _DASH
        v_r2 = _fnum(mejor_r2["r2"] if mejor_r2 else None)
        n_auc = mejor_auc["name"] if mejor_auc else _DASH
        v_auc = _fnum(mejor_auc["auc"] if mejor_auc else None)
        # Los nombres de un modelo y su version ajustada solo se diferencian en
        # el sufijo «(tuned)». Decir «ninguno gana en las dos» y acto seguido
        # nombrar dos cadenas casi identicas se lee como una contradiccion, asi
        # que cuando eso pasa hay que senalar de forma explicita que son filas
        # distintas y donde queda el modelo desplegado.
        if act is not None and act["name"] == n_r2:
            situacion = (
                f"el desplegado ({n_r2}) tiene el mejor R² ({v_r2}) pero NO el mejor AUC, "
                f"que se lo lleva {n_auc} ({v_auc}) —otra fila de la tabla—"
            )
        elif act is not None and act["name"] == n_auc:
            situacion = (
                f"el desplegado ({n_auc}) tiene el mejor AUC ({v_auc}) pero NO el mejor R², "
                f"que se lo lleva {n_r2} ({v_r2}) —otra fila de la tabla—"
            )
        else:
            situacion = (
                f"el mejor R² es de {n_r2} ({v_r2}) y el mejor AUC, de {n_auc} ({v_auc}), "
                "y el desplegado no es ninguno de los dos"
            )
        matiz = (
            "Ninguna fila gana en las dos tareas a la vez, y ahi es donde el criterio "
            f"combinado decide: {situacion}. Repartir 50/50 entre regresion y "
            "clasificacion es una decision de diseno, no un resultado: con otro reparto el "
            "ganador podria cambiar."
        )
    coste = ""
    if mas_rapido:
        coste = (
            f" El mas barato de entrenar es {mas_rapido['name']} "
            f"({_fnum(mas_rapido['train_time_s'], 1)} s), util si el reentrenamiento tuviera "
            "que ser frecuente."
        )
    return (
        f"Los {len(rows)} modelos se comparan sobre el MISMO holdout temporal (2018, sin "
        f"barajar). Fila resaltada = modelo desplegado ({quien}). {matiz}{coste} "
        f"{_CRITERIO} RMSE/MAE/R²/NSE/KGE son la media de los objetivos CO₂ y CH₄ "
        "—el detalle por objetivo esta en `ml/training/resultados_comparativa.csv`— y esa "
        "media esconde un contraste grande: el CH₄ se predice bien y el CO₂ mal, asi que "
        "ninguna fila debe leerse como «calidad del modelo» sin abrir el desglose. El KGE "
        "del NEE no esta definido (media ≈ 0)."
    )


def _lectura_confusion(rows: list[dict]) -> str:
    act = _activo(rows)
    if not act or act.get("tp") is None:
        return (
            "Matrices de confusion de los cinco modelos en el holdout de 2018. Clase "
            "positiva = sumidero (minoritaria, ≈ 32 %). Debajo de cada matriz se muestran "
            "F1 y recall de sumidero: con este desbalance la accuracy no distingue un "
            "modelo que aprende de uno que responde «fuente» siempre."
        )
    tn, fp, fn, tp = (int(act[k] or 0) for k in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    trivial = (tn + fp) / total if total else 0.0
    pct_pos = (tp + fn) / total * 100 if total else 0.0
    return (
        f"Clase positiva = sumidero, la minoritaria ({_fnum(pct_pos, 1)} % de los {total} "
        f"dias del holdout). El modelo desplegado —{act['name']}— detecta {tp} de los "
        f"{tp + fn} dias de captura (recall {_fnum(recall, 2)}) y se le escapan {fn}; los "
        f"{fp} falsos positivos son dias que declara sumidero sin serlo. La comparacion "
        f"honesta no es contra el 50 % sino contra el {_fnum(trivial * 100, 1)} % de accuracy "
        "que lograria un clasificador que respondiera «fuente» siempre: por eso debajo de "
        "cada matriz se reportan F1 y recall de sumidero y no la accuracy. Para uso en "
        "restauracion los falsos positivos son el error caro —sobreestiman el beneficio de "
        "una intervencion— y conviene mirarlos antes que el acierto global."
    )


def _lectura_roc(rows: list[dict]) -> str:
    con_auc = [r for r in rows if r.get("auc") is not None]
    if not con_auc:
        return (
            "Las cinco curvas ROC sobre el mismo eje: separan sumidero de fuente variando "
            "el umbral de decision. La diagonal es el azar."
        )
    peor = min(con_auc, key=lambda r: float(r["auc"]))
    mejor = max(con_auc, key=lambda r: float(r["auc"]))
    act = _activo(rows)
    cola = ""
    if act and act.get("auc") is not None:
        cola = f" El modelo desplegado es {act['name']}, con AUC {_fnum(act['auc'])}."
    return (
        f"Las {len(con_auc)} curvas sobre el mismo eje, de {_fnum(peor['auc'])} "
        f"({peor['name']}) a {_fnum(mejor['auc'])} ({mejor['name']}); la diagonal es el azar. "
        "El AUC resume el rendimiento a TODOS los umbrales, asi que mide la capacidad de "
        "ordenar dias por probabilidad de sumidero, no el acierto con el umbral 0.5 que usa "
        "la matriz de confusion. La distancia entre la mejor y la peor curva es de "
        f"{_fnum(float(mejor['auc']) - float(peor['auc']))} de AUC: un margen estrecho, y por "
        "eso el paso de pruebas estadisticas comprueba si esas diferencias son "
        "significativas o caben dentro de la variabilidad entre folds." + cola
    )


def _lectura_heatmap(rows: list[dict]) -> str:
    return (
        f"Cada columna se normaliza min-max entre los {len(rows)} modelos: verde = mejor de "
        "la columna, rojo = peor. Lo que la figura muestra es el ORDEN relativo, no la "
        "magnitud —un verde intenso puede corresponder a un R² mediocre si todos los "
        "modelos son mediocres en esa metrica, como ocurre con el NEE— asi que se lee "
        "junto a la tabla de arriba y nunca sola. Su utilidad es ver de un vistazo si algun "
        "modelo domina en bloque o si cada uno gana en una columna distinta, que es justo lo "
        "que obliga a fijar un criterio de seleccion explicito."
    )


def _explicabilidad_confusion(rows: list[dict]) -> str:
    n = len(rows)
    return (
        f"Una matriz de confusion por cada uno de los {n} modelos, todas sobre el mismo "
        "holdout de 2018. Las filas son la clase real y las columnas la predicha, asi "
        "que la diagonal son los aciertos. La clase positiva es «sumidero», definida "
        "como NEE < 0. Debajo de cada matriz se imprimen el F1 y el recall de esa clase."
    )


def _explicabilidad_roc(rows: list[dict]) -> str:
    n = len([r for r in rows if r.get("auc") is not None])
    return (
        f"Las {n} curvas ROC superpuestas sobre unos mismos ejes: tasa de falsos "
        "positivos en X frente a tasa de verdaderos positivos (recall de sumidero) en Y, "
        "barriendo todos los umbrales de decision. La diagonal representa al "
        "clasificador aleatorio y el AUC de la leyenda es el area bajo cada curva."
    )


def _explicabilidad_heatmap(rows: list[dict]) -> str:
    return (
        f"Una fila por modelo y una columna por metrica. Cada COLUMNA se normaliza "
        f"min-max por separado entre los {len(rows)} modelos, de modo que el mejor valor "
        "de esa metrica queda en verde y el peor en rojo sea cual sea su magnitud real; "
        "en las metricas de error el sentido se invierte antes de colorear para que "
        "verde signifique siempre «mejor». Los numeros de las celdas son los valores ya "
        "normalizados, no las metricas originales."
    )


FIGURES = [
    (
        "matrices_confusion.png",
        "Matrices de confusion (holdout 2018)",
        _explicabilidad_confusion,
        _lectura_confusion,
    ),
    (
        "curvas_roc_comparacion.png",
        "Curvas ROC — sumidero vs fuente",
        _explicabilidad_roc,
        _lectura_roc,
    ),
    (
        "heatmap_comparativo_metricas.png",
        "Heatmap comparativo de metricas",
        _explicabilidad_heatmap,
        _lectura_heatmap,
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

    I.titulo("Tabla comparativa")
    I.tabla(
        df.style.apply(_highlight_active, axis=1, active_names=active_names),
        explicabilidad=(
            "Una fila por modelo registrado en `ml_models`, con sus metricas del holdout "
            "de 2018 leidas de `ml_model_metrics`. RMSE y MAE son errores en la unidad "
            "del objetivo (menor es mejor); R², NSE y KGE son adimensionales y valen 1 "
            "en la prediccion perfecta; Accuracy, F1 y AUC son de la clasificacion "
            "sumidero/fuente y van de 0 a 1. «t entren.» son los segundos que costo "
            "ajustar el modelo. La fila con fondo verde es la version desplegada."
        ),
        formato={
            "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}",
            "NSE": "{:.3f}", "KGE": "{:.3f}", "Accuracy": "{:.3f}",
            "F1": "{:.3f}", "AUC": "{:.3f}", "t entren. (s)": "{:.1f}",
        },
        interpretacion=_lectura_tabla(rows, active_names),
    )

    I.titulo("Figuras del estudio (Pasos 5–6)")
    cols = st.columns(2)
    for i, (filename, title, explicabilidad_fn, lectura) in enumerate(FIGURES):
        with cols[i % 2]:
            I.figura(
                C.TRAIN_FIG / filename,
                titulo_fig=title,
                explicabilidad=explicabilidad_fn(rows),
                interpretacion=lectura(rows),
                comando="python -m ml.training.run_training",
            )
