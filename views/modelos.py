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
from ui.i18n import force, lang, tr

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
    quien = ", ".join(sorted(active_names)) or tr("ninguno", "none")
    coincide = (
        act is not None and mejor_r2 is not None and mejor_auc is not None
        and act["name"] == mejor_r2["name"] == mejor_auc["name"]
    )
    if coincide:
        matiz = tr(
            "El modelo desplegado gana a la vez en regresion y en clasificacion, asi que el "
            "criterio combinado no arbitra nada: la eleccion habria sido la misma con "
            "cualquiera de los dos pesos.",
            "The deployed model wins on regression and classification at once, so the "
            "combined criterion arbitrates nothing: the choice would have been the same "
            "with either weighting.",
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
            situacion = tr(
                f"el desplegado ({n_r2}) tiene el mejor R² ({v_r2}) pero NO el mejor AUC, "
                f"que se lo lleva {n_auc} ({v_auc}) —otra fila de la tabla—",
                f"the deployed one ({n_r2}) has the best R² ({v_r2}) but NOT the best AUC, "
                f"which goes to {n_auc} ({v_auc}) —another row of the table—",
            )
        elif act is not None and act["name"] == n_auc:
            situacion = tr(
                f"el desplegado ({n_auc}) tiene el mejor AUC ({v_auc}) pero NO el mejor R², "
                f"que se lo lleva {n_r2} ({v_r2}) —otra fila de la tabla—",
                f"the deployed one ({n_auc}) has the best AUC ({v_auc}) but NOT the best R², "
                f"which goes to {n_r2} ({v_r2}) —another row of the table—",
            )
        else:
            situacion = tr(
                f"el mejor R² es de {n_r2} ({v_r2}) y el mejor AUC, de {n_auc} ({v_auc}), "
                "y el desplegado no es ninguno de los dos",
                f"the best R² belongs to {n_r2} ({v_r2}) and the best AUC to {n_auc} "
                f"({v_auc}), and the deployed one is neither",
            )
        matiz = tr(
            "Ninguna fila gana en las dos tareas a la vez, y ahi es donde el criterio "
            f"combinado decide: {situacion}. Repartir 50/50 entre regresion y "
            "clasificacion es una decision de diseno, no un resultado: con otro reparto el "
            "ganador podria cambiar.",
            "No row wins both tasks at once, and that is where the combined criterion "
            f"decides: {situacion}. Splitting 50/50 between regression and classification "
            "is a design decision, not a result: with another split the winner could change.",
        )
    coste = ""
    if mas_rapido:
        coste = tr(
            f" El mas barato de entrenar es {mas_rapido['name']} "
            f"({_fnum(mas_rapido['train_time_s'], 1)} s), util si el reentrenamiento tuviera "
            "que ser frecuente.",
            f" The cheapest to train is {mas_rapido['name']} "
            f"({_fnum(mas_rapido['train_time_s'], 1)} s), useful if retraining had to be "
            "frequent.",
        )
    return tr(
        f"Los {len(rows)} modelos se comparan sobre el MISMO holdout temporal (2018, sin "
        f"barajar). Fila resaltada = modelo desplegado ({quien}). {matiz}{coste} "
        f"{_criterio()} RMSE/MAE/R²/NSE/KGE son la media de los objetivos CO₂ y CH₄ "
        "—el detalle por objetivo esta en `ml/training/resultados_comparativa.csv`— y esa "
        "media esconde un contraste grande: el CH₄ se predice bien y el CO₂ mal, asi que "
        "ninguna fila debe leerse como «calidad del modelo» sin abrir el desglose. El KGE "
        "del NEE no esta definido (media ≈ 0).",
        f"The {len(rows)} models are compared on the SAME temporal holdout (2018, "
        f"unshuffled). Highlighted row = deployed model ({quien}). {matiz}{coste} "
        f"{_criterio()} RMSE/MAE/R²/NSE/KGE are the mean over the CO₂ and CH₄ targets "
        "—the per-target detail is in `ml/training/resultados_comparativa.csv`— and that "
        "mean hides a large contrast: CH₄ is predicted well and CO₂ poorly, so no row "
        "should be read as «model quality» without opening the breakdown. The NEE KGE "
        "is undefined (mean ≈ 0).",
    )


def _lectura_confusion(rows: list[dict]) -> str:
    act = _activo(rows)
    if not act or act.get("tp") is None:
        return tr(
            "Matrices de confusion de los cinco modelos en el holdout de 2018. Clase "
            "positiva = sumidero (minoritaria, ≈ 32 %). Debajo de cada matriz se muestran "
            "F1 y recall de sumidero: con este desbalance la accuracy no distingue un "
            "modelo que aprende de uno que responde «fuente» siempre.",
            "Confusion matrices of the five models on the 2018 holdout. Positive class = "
            "sink (minority, ≈ 32 %). Below each matrix the sink F1 and recall are shown: "
            "with this imbalance, accuracy cannot tell a model that learns from one that "
            "always answers «source».",
        )
    tn, fp, fn, tp = (int(act[k] or 0) for k in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    trivial = (tn + fp) / total if total else 0.0
    pct_pos = (tp + fn) / total * 100 if total else 0.0
    return tr(
        f"Clase positiva = sumidero, la minoritaria ({_fnum(pct_pos, 1)} % de los {total} "
        f"dias del holdout). El modelo desplegado —{act['name']}— detecta {tp} de los "
        f"{tp + fn} dias de captura (recall {_fnum(recall, 2)}) y se le escapan {fn}; los "
        f"{fp} falsos positivos son dias que declara sumidero sin serlo. La comparacion "
        f"honesta no es contra el 50 % sino contra el {_fnum(trivial * 100, 1)} % de accuracy "
        "que lograria un clasificador que respondiera «fuente» siempre: por eso debajo de "
        "cada matriz se reportan F1 y recall de sumidero y no la accuracy. Para uso en "
        "restauracion los falsos positivos son el error caro —sobreestiman el beneficio de "
        "una intervencion— y conviene mirarlos antes que el acierto global.",
        f"Positive class = sink, the minority ({_fnum(pct_pos, 1)} % of the {total} holdout "
        f"days). The deployed model —{act['name']}— detects {tp} of the {tp + fn} uptake "
        f"days (recall {_fnum(recall, 2)}) and misses {fn}; the {fp} false positives are "
        "days it declares sink without being so. The honest comparison is not against "
        f"50 % but against the {_fnum(trivial * 100, 1)} % accuracy that a classifier "
        "always answering «source» would reach: that is why sink F1 and recall, not "
        "accuracy, are reported below each matrix. For restoration use, false positives "
        "are the costly error —they overestimate the benefit of an intervention— and "
        "should be checked before overall accuracy.",
    )


def _lectura_roc(rows: list[dict]) -> str:
    con_auc = [r for r in rows if r.get("auc") is not None]
    if not con_auc:
        return tr(
            "Las cinco curvas ROC sobre el mismo eje: separan sumidero de fuente variando "
            "el umbral de decision. La diagonal es el azar.",
            "The five ROC curves on the same axes: they separate sink from source by "
            "varying the decision threshold. The diagonal is chance.",
        )
    peor = min(con_auc, key=lambda r: float(r["auc"]))
    mejor = max(con_auc, key=lambda r: float(r["auc"]))
    act = _activo(rows)
    cola = ""
    if act and act.get("auc") is not None:
        cola = tr(
            f" El modelo desplegado es {act['name']}, con AUC {_fnum(act['auc'])}.",
            f" The deployed model is {act['name']}, with AUC {_fnum(act['auc'])}.",
        )
    return tr(
        f"Las {len(con_auc)} curvas sobre el mismo eje, de {_fnum(peor['auc'])} "
        f"({peor['name']}) a {_fnum(mejor['auc'])} ({mejor['name']}); la diagonal es el azar. "
        "El AUC resume el rendimiento a TODOS los umbrales, asi que mide la capacidad de "
        "ordenar dias por probabilidad de sumidero, no el acierto con el umbral 0.5 que usa "
        "la matriz de confusion. La distancia entre la mejor y la peor curva es de "
        f"{_fnum(float(mejor['auc']) - float(peor['auc']))} de AUC: un margen estrecho, y por "
        "eso el paso de pruebas estadisticas comprueba si esas diferencias son "
        "significativas o caben dentro de la variabilidad entre folds." + cola,
        f"The {len(con_auc)} curves on the same axes, from {_fnum(peor['auc'])} "
        f"({peor['name']}) to {_fnum(mejor['auc'])} ({mejor['name']}); the diagonal is "
        "chance. AUC summarizes performance at ALL thresholds, so it measures the ability "
        "to rank days by sink probability, not the accuracy at the 0.5 threshold used by "
        "the confusion matrix. The gap between the best and the worst curve is "
        f"{_fnum(float(mejor['auc']) - float(peor['auc']))} AUC: a narrow margin, which is "
        "why the statistical-tests step checks whether those differences are significant "
        "or fit within the variability between folds." + cola,
    )


def _lectura_heatmap(rows: list[dict]) -> str:
    return tr(
        f"Cada columna se normaliza min-max entre los {len(rows)} modelos: verde = mejor de "
        "la columna, rojo = peor. Lo que la figura muestra es el ORDEN relativo, no la "
        "magnitud —un verde intenso puede corresponder a un R² mediocre si todos los "
        "modelos son mediocres en esa metrica, como ocurre con el NEE— asi que se lee "
        "junto a la tabla de arriba y nunca sola. Su utilidad es ver de un vistazo si algun "
        "modelo domina en bloque o si cada uno gana en una columna distinta, que es justo lo "
        "que obliga a fijar un criterio de seleccion explicito.",
        f"Each column is min-max normalized across the {len(rows)} models: green = best "
        "of the column, red = worst. What the figure shows is the relative ORDER, not the "
        "magnitude —an intense green can correspond to a mediocre R² if all models are "
        "mediocre on that metric, as happens with NEE— so it is read together with the "
        "table above and never alone. Its use is to see at a glance whether some model "
        "dominates as a block or whether each wins a different column, which is exactly "
        "what forces an explicit selection criterion.",
    )


def _explicabilidad_confusion(rows: list[dict]) -> str:
    n = len(rows)
    return tr(
        f"Una matriz de confusion por cada uno de los {n} modelos, todas sobre el mismo "
        "holdout de 2018. Las filas son la clase real y las columnas la predicha, asi "
        "que la diagonal son los aciertos. La clase positiva es «sumidero», definida "
        "como NEE < 0. Debajo de cada matriz se imprimen el F1 y el recall de esa clase.",
        f"One confusion matrix for each of the {n} models, all on the same 2018 holdout. "
        "Rows are the true class and columns the predicted one, so the diagonal holds "
        "the hits. The positive class is «sink», defined as NEE < 0. Below each matrix "
        "the F1 and recall of that class are printed.",
    )


def _explicabilidad_roc(rows: list[dict]) -> str:
    n = len([r for r in rows if r.get("auc") is not None])
    return tr(
        f"Las {n} curvas ROC superpuestas sobre unos mismos ejes: tasa de falsos "
        "positivos en X frente a tasa de verdaderos positivos (recall de sumidero) en Y, "
        "barriendo todos los umbrales de decision. La diagonal representa al "
        "clasificador aleatorio y el AUC de la leyenda es el area bajo cada curva.",
        f"The {n} ROC curves overlaid on the same axes: false positive rate on X against "
        "true positive rate (sink recall) on Y, sweeping all decision thresholds. The "
        "diagonal represents the random classifier and the AUC in the legend is the area "
        "under each curve.",
    )


def _explicabilidad_heatmap(rows: list[dict]) -> str:
    return tr(
        f"Una fila por modelo y una columna por metrica. Cada COLUMNA se normaliza "
        f"min-max por separado entre los {len(rows)} modelos, de modo que el mejor valor "
        "de esa metrica queda en verde y el peor en rojo sea cual sea su magnitud real; "
        "en las metricas de error el sentido se invierte antes de colorear para que "
        "verde signifique siempre «mejor». Los numeros de las celdas son los valores ya "
        "normalizados, no las metricas originales.",
        f"One row per model and one column per metric. Each COLUMN is min-max normalized "
        f"separately across the {len(rows)} models, so the best value of that metric is "
        "green and the worst red regardless of its real magnitude; for error metrics the "
        "direction is inverted before coloring so that green always means «better». The "
        "numbers in the cells are the normalized values, not the original metrics.",
    )


def _figures() -> list[tuple]:
    return [
        (
            "matrices_confusion.png",
            tr("Matrices de confusion (holdout 2018)", "Confusion matrices (2018 holdout)"),
            _explicabilidad_confusion,
            _lectura_confusion,
        ),
        (
            "curvas_roc_comparacion.png",
            tr("Curvas ROC — sumidero vs fuente", "ROC curves — sink vs source"),
            _explicabilidad_roc,
            _lectura_roc,
        ),
        (
            "heatmap_comparativo_metricas.png",
            tr("Heatmap comparativo de metricas", "Comparative metrics heatmap"),
            _explicabilidad_heatmap,
            _lectura_heatmap,
        ),
    ]


def _criterio() -> str:
    return tr(
        "Criterio de seleccion: `combined = 0.5·reg_score_norm + 0.5·(0.5·F1_macro + 0.5·AUC)` "
        "— 50 % regresion (R² medio de CO₂ y CH₄, normalizado min-max entre modelos) y 50 % "
        "clasificacion (F1_macro y AUC, no accuracy, por el desbalance de clases).",
        "Selection criterion: `combined = 0.5·reg_score_norm + 0.5·(0.5·F1_macro + 0.5·AUC)` "
        "— 50 % regression (mean R² of CO₂ and CH₄, min-max normalized across models) and "
        "50 % classification (F1_macro and AUC, not accuracy, because of the class "
        "imbalance).",
    )


def _columns() -> dict[str, str]:
    return {
        "name": tr("Modelo", "Model"),
        "familia": tr("Familia", "Family"),
        "version": tr("Version", "Version"),
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "nse": "NSE",
        "kge": "KGE",
        "accuracy": "Accuracy",
        "f1": "F1",
        "auc": "AUC",
        "train_time_s": tr("t entren. (s)", "train t (s)"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def _comparison() -> list[dict]:
    with session_scope() as db:
        rows = [asdict(r) for r in C.model_rows(db)]
    for r in rows:
        r["familia"] = "hybrid" if r["algorithm_type"] == "HYBRID" else "classical"
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
    col = _columns()["name"]
    return [style if row[col] in active_names else "" for _ in row]


_PDF_MIME = "application/pdf"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@st.cache_data(ttl=600, show_spinner=False)
def build_report(kind: str, site_id: int, language: str = "es", **kwargs) -> bytes:
    """Genera un informe bajo demanda. Los tres builders devuelven `bytes`.

    `language` forma parte de la clave de cache: sin el, cambiar de idioma
    devolveria el informe ya generado en el otro.
    """
    with force(language), session_scope() as db:
        if kind == "pdf":
            return pdf_report.build(db, site_id=site_id, start=None, end=None)
        if kind == "docx":
            return docx_report.build(db, site_id=site_id, **kwargs)
        return xlsx_report.build(db, site_id=site_id)


def _report_button(kind: str, label: str, filename: str, mime: str, site_id: int, **kwargs) -> None:
    """Dos pasos: generar (caro) y luego descargar, para no penalizar cada rerun."""
    state_key = f"report_{kind}_{site_id}_{lang()}"
    if st.button(tr(f"Generar {label}", f"Generate {label}"), key=f"gen_{state_key}", width="stretch"):
        try:
            with st.spinner(tr(f"Generando {label}…", f"Generating {label}…")):
                st.session_state[state_key] = build_report(kind, site_id, lang(), **kwargs)
        except DomainError as exc:
            st.session_state.pop(state_key, None)
            st.error(str(exc))
    content = st.session_state.get(state_key)
    if content:
        st.download_button(
            tr(f"Descargar {label}", f"Download {label}"),
            content,
            file_name=filename,
            mime=mime,
            key=f"dl_{state_key}",
            type="primary",
            width="stretch",
        )


def _reports_section(site: dict) -> None:
    st.subheader(tr("Informes descargables", "Downloadable reports"))
    st.caption(tr(
        "Se generan al vuelo desde la base de datos y los artefactos de `ml/`. "
        "Cada tabla y figura lleva sus bloques de interpretacion y explicabilidad. La "
        "primera generacion tarda unos segundos porque ejecuta el modelo sobre la serie. "
        "Los informes salen en el idioma activo.",
        "They are generated on the fly from the database and the `ml/` artifacts. "
        "Every table and figure carries its interpretation and explainability blocks. "
        "The first generation takes a few seconds because it runs the model over the "
        "series. Reports come out in the active language.",
    ))
    slug = _slug(site)
    col_pdf, col_docx, col_xlsx = st.columns(3)
    with col_pdf:
        _report_button("pdf", tr("resumen ejecutivo (.pdf)", "executive summary (.pdf)"),
                       f"peattwin_{tr('resumen', 'summary')}_{slug}.pdf", _PDF_MIME, site["id"])
    with col_docx:
        _report_button("docx", tr("informe tecnico (.docx)", "technical report (.docx)"),
                       f"peattwin_{tr('informe_tecnico', 'technical_report')}_{slug}.docx",
                       _DOCX_MIME, site["id"], folds=5)
    with col_xlsx:
        _report_button("xlsx", tr("datos y metricas (.xlsx)", "data and metrics (.xlsx)"),
                       f"peattwin_{tr('datos', 'data')}_{slug}.xlsx", _XLSX_MIME, site["id"])


def render() -> None:
    st.title(tr("Comparacion de modelos (Pasos 5–8)", "Model comparison (Steps 5–8)"))
    st.caption(tr(
        "Metricas de regresion (media de CO₂ y CH₄) y de clasificacion en el holdout "
        "de 2018. La fila resaltada es el modelo desplegado.",
        "Regression metrics (mean of CO₂ and CH₄) and classification metrics on the 2018 "
        "holdout. The highlighted row is the deployed model.",
    ))

    sites = _sites()
    if sites:
        if len(sites) > 1:
            site = st.selectbox(
                tr("Sitio de los informes", "Report site"),
                sites,
                format_func=lambda s: f"{s['name']} · {s['fluxnet_id']}" if s["fluxnet_id"] else s["name"],
            )
        else:
            site = sites[0]
        _reports_section(site)
        st.divider()

    rows = _comparison()
    if not rows:
        st.warning(tr(
            "No hay modelos registrados. Ejecuta `python -m ml.training.run_training` "
            "para entrenar los 5 modelos y poblar `ml_models` / `ml_model_metrics`.",
            "No models are registered. Run `python -m ml.training.run_training` to train "
            "the 5 models and populate `ml_models` / `ml_model_metrics`.",
        ))
        return

    active_names = {r["name"] for r in rows if r["is_active"]}
    columns = _columns()
    df = pd.DataFrame(rows)[list(columns)].rename(columns=columns)
    t_col = columns["train_time_s"]

    I.titulo(tr("Tabla comparativa", "Comparison table"))
    I.tabla(
        df.style.apply(_highlight_active, axis=1, active_names=active_names),
        explicabilidad=tr(
            "Una fila por modelo registrado en `ml_models`, con sus metricas del holdout "
            "de 2018 leidas de `ml_model_metrics`. RMSE y MAE son errores en la unidad "
            "del objetivo (menor es mejor); R², NSE y KGE son adimensionales y valen 1 "
            "en la prediccion perfecta; Accuracy, F1 y AUC son de la clasificacion "
            "sumidero/fuente y van de 0 a 1. «t entren.» son los segundos que costo "
            "ajustar el modelo. La fila con fondo verde es la version desplegada.",
            "One row per model registered in `ml_models`, with its 2018 holdout metrics "
            "read from `ml_model_metrics`. RMSE and MAE are errors in the target's unit "
            "(lower is better); R², NSE and KGE are dimensionless and equal 1 for a "
            "perfect prediction; Accuracy, F1 and AUC belong to the sink/source "
            "classification and range from 0 to 1. «train t» is the seconds it took to "
            "fit the model. The green-background row is the deployed version.",
        ),
        formato={
            "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}",
            "NSE": "{:.3f}", "KGE": "{:.3f}", "Accuracy": "{:.3f}",
            "F1": "{:.3f}", "AUC": "{:.3f}", t_col: "{:.1f}",
        },
        interpretacion=_lectura_tabla(rows, active_names),
    )

    I.titulo(tr("Figuras del estudio (Pasos 5–6)", "Study figures (Steps 5–6)"))
    cols = st.columns(2)
    for i, (filename, title, explicabilidad_fn, lectura) in enumerate(_figures()):
        with cols[i % 2]:
            I.figura(
                C.TRAIN_FIG / filename,
                titulo_fig=title,
                explicabilidad=explicabilidad_fn(rows),
                interpretacion=lectura(rows),
                comando="python -m ml.training.run_training",
            )
