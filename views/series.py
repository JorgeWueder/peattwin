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
from ui import interpret as I
from ui.charts import series_chart
from ui.db import session_scope
from ui.i18n import tr

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


@st.cache_data(ttl=300, show_spinner=False)
def _series(site_id: int, start: date, end: date) -> dict:
    with session_scope() as db:
        resp = model_service.predicted_series(db, site_id=site_id, start=start, end=end)
    return {
        "model_name": resp.model_name,
        "model_version": resp.model_version,
        "points": [p.model_dump() for p in resp.points],
    }


# ---------------------------------------------------------------- interpretaciones
# Se calculan sobre el rango que el usuario tiene delante, no sobre constantes
# del holdout de 2018: el selector de fechas cambia la serie, y una lectura fija
# ("R2 = 0.24") dejaria de describir el grafico en cuanto se mueve el rango.

def _calidad(r2: float) -> str:
    if r2 >= 0.75:
        return tr("sigue bien la serie", "tracks the series well")
    if r2 >= 0.5:
        return tr("capta la forma general pero pierde amplitud",
                  "captures the overall shape but loses amplitude")
    if r2 >= 0.25:
        return tr("solo reproduce el ciclo estacional, no el dia a dia",
                  "only reproduces the seasonal cycle, not the day-to-day")
    return tr("apenas mejora a predecir la media del periodo",
              "barely improves on predicting the period mean")


def _explicabilidad_metricas(n_dias: int, inicio, fin) -> str:
    return tr(
        f"Resumen del ajuste sobre los {n_dias} dias del rango {inicio}–{fin}. RMSE es "
        "el error cuadratico medio en la unidad de cada flujo (gC m⁻² d⁻¹ para el CO₂, "
        "nmol m⁻² s⁻¹ para el CH₄): menor es mejor y 0 seria perfecto. R² es la "
        "fraccion de varianza explicada, adimensional, con 1 = perfecto y 0 = tan bueno "
        "como predecir siempre la media. «Clase acierto» es el porcentaje de dias en que "
        "coinciden la clase observada y la predicha (sumidero si el NEE es negativo).",
        f"Fit summary over the {n_dias} days in the range {inicio}–{fin}. RMSE is the "
        "root mean squared error in each flux's unit (gC m⁻² d⁻¹ for CO₂, "
        "nmol m⁻² s⁻¹ for CH₄): lower is better and 0 would be perfect. R² is the "
        "fraction of variance explained, dimensionless, with 1 = perfect and 0 = no better "
        "than always predicting the mean. «Class accuracy» is the percentage of days on "
        "which the observed and predicted classes agree (sink if NEE is negative).",
    )


def _explicabilidad_serie(unidad: str, signo: str) -> str:
    return tr(
        f"Serie diaria del rango elegido: en negro los valores observados por la torre "
        f"de covarianza turbulenta y en verde los que predice el modelo activo, ambos en "
        f"{unidad}. El eje X son fechas reales, no indices. Es un BACKTEST: el modelo "
        "recibe las variables medidas de cada dia, no la climatologia que usa el "
        f"simulador. {signo}",
        f"Daily series over the chosen range: in black the values observed by the eddy "
        f"covariance tower and in green those predicted by the active model, both in "
        f"{unidad}. The X axis shows real dates, not indices. This is a BACKTEST: the "
        "model receives each day's measured variables, not the climatology used by the "
        f"simulator. {signo}",
    )


def _lectura_metricas(n_dias: int, co2_r2: float, ch4_r2: float, acc: float) -> str:
    return tr(
        f"Metricas recalculadas sobre los {n_dias} dias del rango elegido, no sobre el "
        f"holdout completo. El contraste entre R² de CH₄ ({F.fmt(ch4_r2, 2)}) y de CO₂ "
        f"({F.fmt(co2_r2, 2)}) es el resultado central del gemelo: el metano depende de "
        "temperatura de suelo y nivel freatico, ambos medidos y en el dataset, mientras que "
        "el NEE es la diferencia entre fotosintesis y respiracion y depende del estado de la "
        "vegetacion, que el sitio no mide (sin NDVI ni LAI). El acierto de clase "
        f"({F.fmt_pct(acc)}) hay que leerlo contra el 68 % que obtendria un clasificador "
        "trivial de «siempre fuente», por el desbalance 32/68 de la clase sumidero.",
        f"Metrics recomputed over the {n_dias} days in the chosen range, not over the full "
        f"holdout. The contrast between the R² of CH₄ ({F.fmt(ch4_r2, 2)}) and of CO₂ "
        f"({F.fmt(co2_r2, 2)}) is the twin's central result: methane depends on soil "
        "temperature and water table depth, both measured and in the dataset, whereas "
        "NEE is the difference between photosynthesis and respiration and depends on the "
        "state of the vegetation, which the site does not measure (no NDVI or LAI). The "
        f"class accuracy ({F.fmt_pct(acc)}) must be read against the 68 % that a trivial "
        "«always source» classifier would obtain, because of the 32/68 imbalance of the "
        "sink class.",
    )


def _lectura_nee(df: pd.DataFrame, r2: float, rmse: float) -> str:
    obs = df["nee_observed"]
    dias_sumidero = int((obs < 0).sum())
    frac = dias_sumidero / len(obs) if len(obs) else 0.0
    amplitud_obs = float(obs.max() - obs.min())
    amplitud_pred = float(df["nee_predicted"].max() - df["nee_predicted"].min())
    compresion = 1 - amplitud_pred / amplitud_obs if amplitud_obs else 0.0
    return tr(
        f"Por debajo de la linea de cero el sitio captura carbono: ocurre {dias_sumidero} de "
        f"{len(obs)} dias ({F.fmt_pct(frac)}), concentrados en verano. La curva predicha "
        f"recorre un {F.fmt_pct(compresion)} menos de amplitud que la observada, que es la "
        f"firma visual de un R² de {F.fmt(r2, 2)}: el modelo {_calidad(r2)} y amortigua los "
        f"picos diarios. Con RMSE = {F.fmt(rmse, 2)} gC m⁻² d⁻¹ el error tipico es del "
        "orden de la propia desviacion diaria del NEE, asi que esta serie sirve para el "
        "balance estacional y para clasificar sumidero/fuente, no para el dato de un dia "
        "concreto.",
        f"Below the zero line the site captures carbon: this happens on {dias_sumidero} of "
        f"{len(obs)} days ({F.fmt_pct(frac)}), concentrated in summer. The predicted curve "
        f"spans {F.fmt_pct(compresion)} less amplitude than the observed one, which is the "
        f"visual signature of an R² of {F.fmt(r2, 2)}: the model {_calidad(r2)} and dampens "
        f"the daily peaks. With RMSE = {F.fmt(rmse, 2)} gC m⁻² d⁻¹ the typical error is of "
        "the order of the daily NEE deviation itself, so this series is useful for the "
        "seasonal balance and for sink/source classification, not for the value of any "
        "single day.",
    )


def _lectura_fch4(df: pd.DataFrame, r2: float, rmse: float) -> str:
    obs = df["fch4_observed"]
    pico_obs = float(obs.max())
    pico_pred = float(df["fch4_predicted"].max())
    media = float(obs.mean())
    return tr(
        f"Flujo siempre positivo: la turbera emite metano todos los dias del rango, con media "
        f"observada de {F.fmt(media, 1)} y pico de {F.fmt(pico_obs, 1)} nmol m⁻² s⁻¹ "
        f"frente a {F.fmt(pico_pred, 1)} del modelo. Con R² = {F.fmt(r2, 2)} el modelo "
        f"{_calidad(r2)}, incluidos los pulsos estivales, y el RMSE de {F.fmt(rmse, 1)} "
        "nmol m⁻² s⁻¹ se concentra en la altura de esos pulsos, no en el nivel de base. "
        "Es la mitad fiable del gemelo y la que sostiene el simulador de escenarios: subir el "
        "nivel freatico reduce CO₂ pero encarece el termino de CH₄, y ese termino es el que "
        "aqui esta bien estimado.",
        f"Always-positive flux: the peatland emits methane every day in the range, with an "
        f"observed mean of {F.fmt(media, 1)} and a peak of {F.fmt(pico_obs, 1)} nmol m⁻² s⁻¹ "
        f"against {F.fmt(pico_pred, 1)} from the model. With R² = {F.fmt(r2, 2)} the model "
        f"{_calidad(r2)}, including the summer pulses, and the RMSE of {F.fmt(rmse, 1)} "
        "nmol m⁻² s⁻¹ is concentrated in the height of those pulses, not in the baseline. "
        "It is the reliable half of the twin and the one that underpins the scenario "
        "simulator: raising the water table reduces CO₂ but raises the CH₄ term, and that "
        "term is the one that is well estimated here.",
    )


def render() -> None:
    st.title(tr("Series temporales — observado vs predicho", "Time series — observed vs predicted"))
    st.caption(tr(
        "CO₂ (NEE) y CH₄ (FCH4) medidos frente a la prediccion del modelo activo, "
        "usando las variables reales de cada dia (backtest).",
        "Measured CO₂ (NEE) and CH₄ (FCH4) against the active model's prediction, "
        "using each day's real variables (backtest).",
    ))

    sites = _sites()
    if not sites:
        st.warning(tr(
            "No hay sitios en la base de datos. Ejecuta `python -m ml.etl.load_to_db` "
            "para cargar DE-Zrk y sus observaciones.",
            "There are no sites in the database. Run `python -m ml.etl.load_to_db` "
            "to load DE-Zrk and its observations.",
        ))
        return

    col_site, col_start, col_end = st.columns([2, 1, 1])
    with col_site:
        site = st.selectbox(
            tr("Sitio", "Site"),
            sites,
            format_func=lambda s: f"{s['name']} · {s['fluxnet_id']}" if s["fluxnet_id"] else s["name"],
        )
    with col_start:
        start = st.date_input(tr("Desde", "From"), DEFAULT_START, format="YYYY-MM-DD")
    with col_end:
        end = st.date_input(tr("Hasta", "To"), DEFAULT_END, format="YYYY-MM-DD")

    if start > end:
        st.error(tr("La fecha inicial es posterior a la final.", "The start date is after the end date."))
        return

    try:
        with st.spinner(tr("Ejecutando el modelo activo sobre la serie…",
                           "Running the active model over the series…")):
            data = _series(site["id"], start, end)
    except DomainError as exc:
        if exc.kind == "not_found":
            st.info(tr(
                f"**Sin serie predicha para {site['name']}.** Solo el sitio "
                "**DE-Zrk (Zarnekow)** tiene un dataset procesado con el que ejecutar "
                "el modelo. Selecciona ese sitio para ver la comparacion observado vs "
                "predicho.",
                f"**No predicted series for {site['name']}.** Only the "
                "**DE-Zrk (Zarnekow)** site has a processed dataset to run the model on. "
                "Select that site to see the observed vs predicted comparison.",
            ))
        else:
            st.error(str(exc))
        return

    points = data["points"]
    if not points:
        st.info(tr("Sin datos en el rango seleccionado.", "No data in the selected range."))
        return

    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"])

    co2_rmse, co2_r2 = regression_scores(df["nee_observed"].tolist(), df["nee_predicted"].tolist())
    ch4_rmse, ch4_r2 = regression_scores(df["fch4_observed"].tolist(), df["fch4_predicted"].tolist())
    acc = float((df["class_observed"] == df["class_predicted"]).mean())

    I.metricas(
        [
            (tr("Dias", "Days"), len(df)),
            ("RMSE CO₂", F.fmt(co2_rmse, 2)),
            ("R² CO₂", F.fmt(co2_r2, 2)),
            ("RMSE CH₄", F.fmt(ch4_rmse, 1)),
            ("R² CH₄", F.fmt(ch4_r2, 2)),
            (tr("Clase acierto", "Class accuracy"), F.fmt_pct(acc)),
        ],
        explicabilidad=_explicabilidad_metricas(len(df), start.isoformat(), end.isoformat()),
        interpretacion=_lectura_metricas(len(df), co2_r2, ch4_r2, acc),
    )

    I.titulo("CO₂ — NEE (gC m⁻² d⁻¹)")
    I.grafico(
        series_chart(
            df.rename(columns={"nee_observed": "observado", "nee_predicted": "predicho"})[
                ["date", "observado", "predicho"]
            ],
            unit="gC m⁻² d⁻¹",
            zero_line=True,
        ),
        explicabilidad=_explicabilidad_serie(
            "gC m⁻² d⁻¹",
            tr("La linea discontinua horizontal marca el cero, que separa captura "
               "(valores negativos) de emision (positivos).",
               "The horizontal dashed line marks zero, which separates uptake "
               "(negative values) from emission (positive)."),
        ),
        interpretacion=_lectura_nee(df, co2_r2, co2_rmse),
    )

    I.titulo("CH₄ — FCH4 (nmol CH₄ m⁻² s⁻¹)")
    I.grafico(
        series_chart(
            df.rename(columns={"fch4_observed": "observado", "fch4_predicted": "predicho"})[
                ["date", "observado", "predicho"]
            ],
            unit="nmol m⁻² s⁻¹",
        ),
        explicabilidad=_explicabilidad_serie(
            "nmol m⁻² s⁻¹",
            tr("No hay linea de cero porque el flujo de metano es siempre positivo: la "
               "turbera emite, nunca absorbe.",
               "There is no zero line because the methane flux is always positive: the "
               "peatland emits, it never absorbs."),
        ),
        interpretacion=_lectura_fch4(df, ch4_r2, ch4_rmse),
    )

    st.caption(tr("Modelo activo", "Active model") + f": **{data['model_name']} · {data['model_version']}**")
