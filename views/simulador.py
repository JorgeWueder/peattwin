"""Simulador de escenarios de restauracion hidrologica.

Mueve el nivel freatico objetivo y muestra la respuesta del ecosistema segun el
modelo activo: NEE (CO₂), FCH4 (CH₄) y la clase sumidero/fuente. La meteo y la
temperatura del suelo se toman de la climatologia del dia del anio.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.core.errors import DomainError
from app.schemas.predict import PredictRequest
from app.schemas.restoration_scenario import RestorationScenarioCreate
from app.services import prediction_service, scenario_service, site_service
from ui import auth
from ui import format as F
from ui import interpret as I
from ui.charts import response_curve_chart
from ui.db import session_scope
from ui.i18n import lang, tr

WTD_MIN, WTD_MAX = -20, 80
CURVE_POINTS = 13
DEFAULT_DATE = date(2019, 7, 15)

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _predict(wtd_cm: float, on_date: date) -> dict:
    with session_scope() as db:
        resp = prediction_service.run(
            db,
            PredictRequest(target_water_table_depth_cm=wtd_cm, date=on_date, persist=False),
        )
    return resp.model_dump()


@st.cache_data(ttl=300, show_spinner=False)
def _curve(on_date: date, _model_key: str) -> pd.DataFrame:
    """CURVE_POINTS simulaciones entre WTD_MIN y WTD_MAX para una misma fecha."""
    step = (WTD_MAX - WTD_MIN) / (CURVE_POINTS - 1)
    xs = [round(WTD_MIN + step * i) for i in range(CURVE_POINTS)]
    rows = []
    with session_scope() as db:
        for x in xs:
            resp = prediction_service.run(
                db,
                PredictRequest(target_water_table_depth_cm=x, date=on_date, persist=False),
            )
            rows.append({"wtd": x, "co2": resp.co2_predicted, "ch4": resp.ch4_predicted})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def _sites() -> list[dict]:
    with session_scope() as db:
        rows = [
            {"id": s.id, "name": s.name, "fluxnet_id": s.fluxnet_id}
            for s in site_service.list_sites(db)
        ]
    rows.sort(key=lambda s: (s["fluxnet_id"] != "DE-Zrk", s["id"]))
    return rows


def _scenarios() -> list[dict]:
    with session_scope() as db:
        return [
            {"id": s.id, "name": s.name, "wtd": s.target_water_table_depth}
            for s in scenario_service.list_scenarios(db)
            if s.target_water_table_depth is not None
        ]


def _save_scenario(site_id: int, name: str, wtd: float, description: str, user_id: int) -> int:
    with session_scope() as db:
        created = scenario_service.create_scenario(
            db,
            RestorationScenarioCreate(
                site_id=site_id,
                name=name,
                target_water_table_depth=wtd,
                description=description or None,
            ),
            created_by_id=user_id,
        )
        return created.id


def _read_only_notice() -> None:
    user = auth.current_user()
    roles = ", ".join(user.role_names) if user and user.role_names else tr("sin rol", "no role")
    st.warning(tr(
        f"**Vista de solo lectura.** Tu rol ({roles}) no tiene el permiso "
        "`predictions:run`, asi que no puede ejecutar simulaciones. Un usuario con rol "
        "**investigador** o **admin** si puede usar este panel. El resto del dashboard "
        "(series y comparacion de modelos) esta disponible para tu rol.",
        f"**Read-only view.** Your role ({roles}) does not have the `predictions:run` "
        "permission, so it cannot run simulations. A user with the **researcher** or "
        "**admin** role can use this panel. The rest of the dashboard (series and model "
        "comparison) is available to your role.",
    ))


# ---------------------------------------------------------------- interpretaciones
# Las curvas de respuesta son el resultado que un lector no modelador viene a
# buscar, asi que la lectura se calcula sobre la curva que esta en pantalla: la
# fecha y el modelo activo la cambian, y un texto fijo dejaria de describirla.

# Conversiones a una unidad comun para poder comparar los dos gases.
#   NEE: gC m-2 d-1 -> g CO2 m-2 d-1   (44.009 / 12.011)
#   FCH4: nmol m-2 s-1 -> g CH4 m-2 d-1 (1e-9 mol * 86400 s * 16.043 g/mol)
# El metano se pondera con GWP100 = 27 (IPCC AR6, metano de origen biogenico).
# Es un SUPUESTO declarado, no una salida del modelo: con horizonte de 20 anios
# (GWP20 ~ 80) el compromiso se desplaza y el rehumedecimiento sale peor a corto
# plazo. El modelo predice flujos; la equivalencia es del lector.
_GC_A_GCO2 = 44.009 / 12.011
_NMOL_A_GCH4 = 1e-9 * 86400 * 16.043
_GWP100_CH4 = 27.0


def _co2eq(co2_gc: float, ch4_nmol: float) -> float:
    """g CO2-eq m-2 d-1 de un punto de la curva (positivo = emision neta)."""
    return co2_gc * _GC_A_GCO2 + ch4_nmol * _NMOL_A_GCH4 * _GWP100_CH4


def _clase(nombre: str) -> str:
    return {"sumidero": tr("sumidero", "sink"), "fuente": tr("fuente", "source")}.get(
        nombre, nombre
    )


def _explicabilidad_prediccion(wtd: int, on_date) -> str:
    return tr(
        f"Salida del modelo activo para un unico escenario: nivel freatico llevado a "
        f"{F.fmt_signed(wtd, 0)} cm el {on_date.isoformat()}. NEE en gC m⁻² d⁻¹ con el "
        "signo negativo indicando captura; FCH4 en nmol m⁻² s⁻¹, siempre positivo; y la "
        "clase de balance que devuelve el clasificador. El resto de drivers "
        "—meteorologia y temperatura de suelo— se toman de la climatologia de ese dia "
        "del ano, asi que el escenario es hipotetico y no corresponde a ningun dia real.",
        f"Output of the active model for a single scenario: water table set to "
        f"{F.fmt_signed(wtd, 0)} cm on {on_date.isoformat()}. NEE in gC m⁻² d⁻¹ with a "
        "negative sign meaning uptake; FCH4 in nmol m⁻² s⁻¹, always positive; and the "
        "balance class returned by the classifier. The remaining drivers "
        "—weather and soil temperature— are taken from the climatology of that day of "
        "the year, so the scenario is hypothetical and matches no real day.",
    )


def _explicabilidad_curva(gas: str, unidad: str) -> str:
    return tr(
        f"Respuesta del {gas} al nivel freatico: {CURVE_POINTS} simulaciones "
        f"independientes entre {WTD_MIN} y +{WTD_MAX} cm para la MISMA fecha, con todos "
        f"los demas drivers congelados en su climatologia. El eje X es el nivel freatico "
        f"objetivo en cm (positivo = lamina de agua sobre la superficie) y el Y, el flujo "
        f"predicho en {unidad}. La linea vertical discontinua marca el valor "
        "seleccionado ahora mismo en el control de la izquierda.",
        f"Response of {gas} to the water table: {CURVE_POINTS} independent simulations "
        f"between {WTD_MIN} and +{WTD_MAX} cm for the SAME date, with all other drivers "
        f"frozen at their climatology. The X axis is the target water table in cm "
        f"(positive = water layer above the surface) and Y is the predicted flux in "
        f"{unidad}. The vertical dashed line marks the value currently selected in the "
        "control on the left.",
    )


def _lectura_prediccion(pred: dict, wtd: int, on_date: date) -> str:
    nee = pred["co2_predicted"]
    ch4 = pred["ch4_predicted"]
    signo = pred["class_from_nee_sign"]
    coincide = signo == pred["predicted_class"]
    total = _co2eq(nee, ch4)
    parte_ch4 = ch4 * _NMOL_A_GCH4 * _GWP100_CH4
    peso = parte_ch4 / abs(total) if total else 0.0
    clase_pred = _clase(pred["predicted_class"])
    clase_signo = _clase(signo)
    if coincide:
        concordancia = tr(
            "El clasificador y el signo del NEE coinciden, asi que la etiqueta es solida.",
            "The classifier and the NEE sign agree, so the label is solid.",
        )
    else:
        concordancia = tr(
            "Atencion: el clasificador dice "
            f"**{clase_pred}** pero el signo del NEE predicho dice "
            f"**{clase_signo}**. Son dos rutas distintas (una cabeza de clasificacion "
            "entrenada con la clase observada frente al signo de la regresion) y su "
            "desacuerdo marca un punto de operacion ambiguo: conviene no apoyar una "
            "decision en el.",
            "Warning: the classifier says "
            f"**{clase_pred}** but the sign of the predicted NEE says "
            f"**{clase_signo}**. These are two different routes (a classification head "
            "trained on the observed class versus the regression sign) and their "
            "disagreement marks an ambiguous operating point: a decision should not "
            "rest on it.",
        )
    return tr(
        f"Escenario: nivel freatico a {F.fmt_signed(wtd, 0)} cm el {on_date.isoformat()}, con "
        "el resto de drivers tomados de la climatologia de ese dia del anio. "
        f"P(sumidero) = {F.fmt(pred['proba_sumidero'], 3)}. {concordancia} "
        f"Sumando los dos gases en una misma unidad, el balance es de "
        f"{F.fmt_signed(total, 2)} g CO₂-eq m⁻² d⁻¹ (GWP100 = 27), y el metano aporta "
        f"{F.fmt_pct(abs(peso))} de esa magnitud: la etiqueta «sumidero», que mira solo al "
        "NEE, no equivale por si sola a un beneficio climatico neto.",
        f"Scenario: water table at {F.fmt_signed(wtd, 0)} cm on {on_date.isoformat()}, with "
        "the remaining drivers taken from the climatology of that day of the year. "
        f"P(sink) = {F.fmt(pred['proba_sumidero'], 3)}. {concordancia} "
        f"Adding both gases in a common unit, the balance is "
        f"{F.fmt_signed(total, 2)} g CO₂-eq m⁻² d⁻¹ (GWP100 = 27), and methane accounts "
        f"for {F.fmt_pct(abs(peso))} of that magnitude: the «sink» label, which looks "
        "only at NEE, is not by itself equivalent to a net climate benefit.",
    )


def _forma(serie: pd.Series) -> str:
    """Describe la forma de la curva sin presuponerla.

    La respuesta al nivel freatico NO es monotona en este sitio: decirlo de
    antemano seria afirmar algo que la figura no muestra.
    """
    if serie.is_monotonic_increasing:
        return tr("crece de forma sostenida", "rises steadily")
    if serie.is_monotonic_decreasing:
        return tr("decrece de forma sostenida", "falls steadily")
    i = int(serie.reset_index(drop=True).idxmax())
    j = int(serie.reset_index(drop=True).idxmin())
    ultimo = len(serie) - 1
    if 0 < i < ultimo:
        return tr("tiene un maximo interior", "has an interior maximum")
    if 0 < j < ultimo:
        return tr("tiene un minimo interior", "has an interior minimum")
    return tr("no sigue una direccion unica", "follows no single direction")


def _lectura_curva_co2(curve: pd.DataFrame, wtd: int) -> str:
    co2 = curve["co2"]
    lo, hi = float(co2.min()), float(co2.max())
    wtd_opt = float(curve.loc[co2.idxmin(), "wtd"])
    rango = hi - lo
    negativos = curve[curve["co2"] < 0]
    positivos = curve[curve["co2"] >= 0]

    if positivos.empty:
        signo = tr(
            "En TODO el rango simulado el NEE predicho es negativo: para esta fecha el "
            "modelo da captura de CO₂ con cualquier nivel freatico, asi que el manejo del "
            "agua no decide aqui el signo del balance, solo su magnitud",
            "Across the WHOLE simulated range the predicted NEE is negative: for this date "
            "the model gives CO₂ uptake at any water table, so water management does not "
            "decide the sign of the balance here, only its magnitude",
        )
    elif negativos.empty:
        signo = tr(
            "En todo el rango simulado el NEE se mantiene positivo: para esta fecha el "
            "modelo no encuentra ningun nivel freatico que convierta el sitio en sumidero",
            "Across the whole simulated range NEE stays positive: for this date the model "
            "finds no water table that turns the site into a sink",
        )
    else:
        cruce = F.fmt_signed(float(negativos["wtd"].min()), 0)
        signo = tr(
            f"El NEE cambia de signo dentro del rango: cruza a captura a partir de {cruce} cm",
            f"NEE changes sign within the range: it crosses to uptake from {cruce} cm",
        )

    borde = float(wtd_opt) in (float(curve["wtd"].min()), float(curve["wtd"].max()))
    matiz_opt = (
        tr(
            " Ese minimo cae en el borde del rango simulado, asi que es un limite de la "
            "simulacion y no un optimo: habria que ampliar el rango para saber donde esta.",
            " That minimum falls at the edge of the simulated range, so it is a limit of "
            "the simulation and not an optimum: the range would have to be widened to "
            "find where it is.",
        )
        if borde
        else ""
    )
    return tr(
        f"{signo}. La curva {_forma(co2)} y su captura maxima esta en "
        f"{F.fmt_signed(wtd_opt, 0)} cm con {F.fmt_signed(lo, 2)} gC m⁻² d⁻¹.{matiz_opt} "
        f"De un extremo a otro el NEE solo se mueve {F.fmt(rango, 2)} gC m⁻² d⁻¹: ese es "
        "todo el margen que el nivel freatico puede mover en el termino de CO₂ para esta "
        "fecha, y conviene compararlo con el error del modelo en NEE antes de darle "
        "significado —con un R² ≈ 0.2 en esta variable, parte de esa variacion puede ser "
        "ruido del modelo y no respuesta del ecosistema. Ademas la curva mueve UNA variable "
        "con el resto congelado en su climatologia: es sensibilidad del modelo, no una "
        "prediccion de lo que ocurriria al rehumedecer de verdad.",
        f"{signo}. The curve {_forma(co2)} and its maximum uptake is at "
        f"{F.fmt_signed(wtd_opt, 0)} cm with {F.fmt_signed(lo, 2)} gC m⁻² d⁻¹.{matiz_opt} "
        f"From one end to the other NEE only moves {F.fmt(rango, 2)} gC m⁻² d⁻¹: that is "
        "the whole margin the water table can move in the CO₂ term for this date, and it "
        "should be compared with the model's NEE error before giving it meaning "
        "—with an R² ≈ 0.2 on this variable, part of that variation may be model noise "
        "and not ecosystem response. Also, the curve moves ONE variable with the rest "
        "frozen at its climatology: it is model sensitivity, not a prediction of what "
        "would happen when actually rewetting.",
    )


def _lectura_curva_ch4(curve: pd.DataFrame, wtd: int) -> str:
    ch4 = curve["ch4"]
    lo, hi = float(ch4.min()), float(ch4.max())
    wtd_max = float(curve.loc[ch4.idxmax(), "wtd"])
    actual = float(curve.iloc[(curve["wtd"] - wtd).abs().idxmin()]["ch4"])
    variacion = (hi - lo) / lo if lo > 0 else float("inf")
    if variacion != float("inf"):
        amplitud = tr(
            f"la emision varia un {F.fmt_pct(variacion)} entre el minimo y el maximo "
            f"({F.fmt(lo, 1)} → {F.fmt(hi, 1)} nmol m⁻² s⁻¹)",
            f"the emission varies by {F.fmt_pct(variacion)} between the minimum and the "
            f"maximum ({F.fmt(lo, 1)} → {F.fmt(hi, 1)} nmol m⁻² s⁻¹)",
        )
    else:
        amplitud = tr(
            f"la emision va de ≈ 0 a {F.fmt(hi, 1)} nmol m⁻² s⁻¹",
            f"the emission goes from ≈ 0 to {F.fmt(hi, 1)} nmol m⁻² s⁻¹",
        )
    if variacion < 0.5:
        contexto = tr(
            " Es una respuesta modesta comparada con el nivel de base: la turbera emite "
            "mucho metano en esta fecha lo haga el nivel freatico lo que haga, y moverlo "
            "reordena poco ese total.",
            " It is a modest response compared with the baseline: the peatland emits a "
            "lot of methane on this date whatever the water table does, and moving it "
            "reshuffles that total very little.",
        )
    else:
        contexto = tr(
            " La respuesta es grande en terminos relativos, asi que el nivel freatico si "
            "es una palanca real sobre el metano en esta fecha.",
            " The response is large in relative terms, so the water table is a real "
            "lever on methane on this date.",
        )
    return tr(
        f"La curva {_forma(ch4)}, con el maximo en {F.fmt_signed(wtd_max, 0)} cm, y "
        f"{amplitud}. En el punto elegido el modelo predice {F.fmt(actual, 1)}.{contexto} "
        "Es el termino del compromiso que se suele omitir cuando se presenta el "
        "rehumedecimiento como beneficio inmediato, y es ademas el que este gemelo estima "
        "con mas fiabilidad (R² ≈ 0.84 en el holdout), porque sus dos drivers "
        "—temperatura de suelo y nivel freatico— estan medidos en el sitio.",
        f"The curve {_forma(ch4)}, with its maximum at {F.fmt_signed(wtd_max, 0)} cm, and "
        f"{amplitud}. At the chosen point the model predicts {F.fmt(actual, 1)}.{contexto} "
        "It is the term of the trade-off usually omitted when rewetting is presented as "
        "an immediate benefit, and it is also the one this twin estimates most reliably "
        "(R² ≈ 0.84 on the holdout), because its two drivers —soil temperature and "
        "water table— are measured on site.",
    )


def _lectura_compromiso(curve: pd.DataFrame, wtd: int) -> str:
    """Lectura conjunta de las dos curvas: donde esta el optimo en CO2-eq."""
    eq = curve.apply(lambda r: _co2eq(r["co2"], r["ch4"]), axis=1)
    mejor = curve.loc[eq.idxmin()]
    actual_idx = (curve["wtd"] - wtd).abs().idxmin()
    delta = float(eq[actual_idx] - eq.min())
    extremo = float(mejor["wtd"]) in (float(curve["wtd"].min()), float(curve["wtd"].max()))
    borde = (
        tr(
            " El optimo cae en el borde del rango simulado, asi que es un limite de la "
            "simulacion y no necesariamente un optimo real: habria que ampliar el rango "
            "para saberlo.",
            " The optimum falls at the edge of the simulated range, so it is a limit of "
            "the simulation and not necessarily a real optimum: the range would have to "
            "be widened to know.",
        )
        if extremo
        else ""
    )
    if delta < 0.005:
        situacion = tr(
            "el punto elegido ya esta en ese minimo",
            "the chosen point is already at that minimum",
        )
    else:
        situacion = tr(
            f"el punto elegido queda {F.fmt(delta, 2)} g CO₂-eq m⁻² d⁻¹ por encima",
            f"the chosen point is {F.fmt(delta, 2)} g CO₂-eq m⁻² d⁻¹ above it",
        )
    wtd_mejor = F.fmt_signed(float(mejor["wtd"]), 0)
    eq_min = F.fmt_signed(float(eq.min()), 2)
    return tr(
        f"**Lectura conjunta de las dos curvas.** Sumando ambos gases con GWP100 = 27, el "
        f"minimo de emision equivalente de esta fecha esta en {wtd_mejor} cm "
        f"({eq_min} g CO₂-eq m⁻² d⁻¹); {situacion}.{borde} "
        "El horizonte temporal "
        "cambia la conclusion: con GWP20 (≈ 80) el peso del metano se triplica y el optimo "
        "se desplaza hacia niveles mas bajos. Por eso la herramienta muestra los dos flujos "
        "por separado y deja la ponderacion explicita en manos de quien decide.",
        f"**Joint reading of the two curves.** Adding both gases with GWP100 = 27, the "
        f"minimum equivalent emission for this date is at {wtd_mejor} cm "
        f"({eq_min} g CO₂-eq m⁻² d⁻¹); {situacion}.{borde} "
        "The time horizon changes the conclusion: with GWP20 (≈ 80) the weight of methane "
        "roughly triples and the optimum shifts towards lower levels. That is why the tool "
        "shows the two fluxes separately and leaves the weighting explicit in the hands "
        "of the decision maker.",
    )


def render() -> None:
    st.title(tr("Simulador de escenarios de restauracion", "Restoration scenario simulator"))
    st.caption(tr(
        "Mueve el nivel freatico objetivo y observa la respuesta del ecosistema: NEE "
        "(CO₂), FCH4 (CH₄) y la clase sumidero/fuente que predice el modelo activo. La "
        "meteo y la temperatura del suelo se toman de la climatologia del dia del anio.",
        "Move the target water table and watch the ecosystem's response: NEE (CO₂), "
        "FCH4 (CH₄) and the sink/source class predicted by the active model. Weather and "
        "soil temperature are taken from the climatology of the day of the year.",
    ))

    if not auth.has_permission("predictions:run"):
        _read_only_notice()
        return

    controles, resultados = st.columns([1, 2], gap="large")

    st.session_state.setdefault("sim_wtd", 20)

    with controles:
        st.subheader(tr("Parametros del escenario", "Scenario parameters"))
        saved = _scenarios()
        if saved:
            options = [None, *saved]
            chosen = st.selectbox(
                tr("Cargar un escenario guardado", "Load a saved scenario"),
                options,
                format_func=lambda s: tr("— ninguno —", "— none —")
                if s is None
                else f"{s['name']} ({F.fmt_signed(s['wtd'], 0)} cm)",
            )
            if chosen is not None and st.session_state.get("_sim_loaded") != chosen["id"]:
                st.session_state["_sim_loaded"] = chosen["id"]
                st.session_state["sim_wtd"] = int(
                    max(WTD_MIN, min(WTD_MAX, round(chosen["wtd"])))
                )
                st.rerun()

        # Sin `value=`: el valor vive en session_state["sim_wtd"], que tambien
        # escribe el selector de escenarios guardados.
        wtd = st.slider(
            tr("Nivel freatico objetivo (cm)", "Target water table (cm)"),
            min_value=WTD_MIN,
            max_value=WTD_MAX,
            step=1,
            key="sim_wtd",
        )
        st.caption(tr(
            "−20 = drenado · 0 = superficie · +80 = inundado. Positivo significa lamina "
            "de agua sobre el suelo: Zarnekow es un fen rehumedecido y opera casi "
            "siempre con valores positivos.",
            "−20 = drained · 0 = surface · +80 = flooded. Positive means a water layer "
            "above the soil: Zarnekow is a rewetted fen and almost always operates at "
            "positive values.",
        ))
        on_date = st.date_input(
            tr("Fecha (define la estacionalidad)", "Date (sets the seasonality)"),
            DEFAULT_DATE,
            format="YYYY-MM-DD",
        )

    try:
        pred = _predict(float(wtd), on_date)
    except DomainError as exc:
        with resultados:
            st.error(str(exc))
        return

    with resultados:
        st.subheader(tr("Prediccion para este escenario", "Prediction for this scenario"))
        st.caption(
            tr("Modelo activo", "Active model")
            + f": **{pred['model_name']} · {pred['model_version']}** "
            + f"({tr('formato', 'format')} {pred['model_format']})"
        )
        I.metricas(
            [
                ("CO₂ · NEE (gC m⁻² d⁻¹)", F.fmt_signed(pred["co2_predicted"], 2)),
                ("CH₄ · FCH4 (nmol m⁻² s⁻¹)", F.fmt(pred["ch4_predicted"], 1)),
                (
                    tr("Balance de carbono", "Carbon balance"),
                    _clase(pred["predicted_class"]).capitalize(),
                ),
            ],
            explicabilidad=_explicabilidad_prediccion(wtd, on_date),
            interpretacion=_lectura_prediccion(pred, wtd, on_date),
        )
        with st.expander(tr("Supuestos de la simulacion", "Simulation assumptions")):
            for a in pred["assumptions"]:
                st.markdown(f"- {a}")

    st.divider()
    I.titulo(
        tr("Curva de respuesta al nivel freatico", "Response curve to the water table"),
        ayuda=tr(
            f"{CURVE_POINTS} simulaciones entre {WTD_MIN} y +{WTD_MAX} cm para la fecha "
            "seleccionada. La linea vertical marca el valor actual.",
            f"{CURVE_POINTS} simulations between {WTD_MIN} and +{WTD_MAX} cm for the "
            "selected date. The vertical line marks the current value.",
        ),
    )
    try:
        curve = _curve(on_date, f"{pred['model_name']}|{pred['model_version']}")
    except DomainError as exc:
        st.error(str(exc))
    else:
        c1, c2 = st.columns(2)
        with c1:
            I.grafico(
                response_curve_chart(curve, metric="co2", marker=wtd, unit="gC m⁻² d⁻¹"),
                explicabilidad=_explicabilidad_curva("CO₂ (NEE)", "gC m⁻² d⁻¹"),
                interpretacion=_lectura_curva_co2(curve, wtd),
            )
        with c2:
            I.grafico(
                response_curve_chart(curve, metric="ch4", marker=wtd, unit="nmol m⁻² s⁻¹"),
                explicabilidad=_explicabilidad_curva("CH₄ (FCH4)", "nmol m⁻² s⁻¹"),
                interpretacion=_lectura_curva_ch4(curve, wtd),
            )
        I.nota(_lectura_compromiso(curve, wtd))

    _save_section(wtd, on_date)


def _save_section(wtd: int, on_date: date) -> None:
    """Guardar el escenario actual y descargar el informe tecnico que lo incluye."""
    if not auth.has_permission("scenarios:write"):
        return
    sites = _sites()
    if not sites:
        return
    site = sites[0]
    user = auth.current_user()

    st.divider()
    st.subheader(tr("Guardar como escenario", "Save as scenario"))
    st.caption(tr(
        f"Se guarda para el sitio {site['name']} con el nivel freatico actual "
        f"({F.fmt_signed(wtd, 0)} cm).",
        f"Saved for site {site['name']} with the current water table "
        f"({F.fmt_signed(wtd, 0)} cm).",
    ))

    with st.form("save_scenario"):
        name = st.text_input(
            tr("Nombre", "Name"),
            placeholder=tr(
                f"Nivel freatico a {F.fmt_signed(wtd, 0)} cm",
                f"Water table at {F.fmt_signed(wtd, 0)} cm",
            ),
        )
        description = st.text_input(tr("Descripcion", "Description"), placeholder=tr("Opcional", "Optional"))
        submitted = st.form_submit_button(tr("Guardar escenario", "Save scenario"))
    if submitted:
        if not name.strip():
            st.error(tr("El escenario necesita un nombre.", "The scenario needs a name."))
        else:
            try:
                new_id = _save_scenario(
                    site["id"], name.strip(), float(wtd), description, user.id
                )
            except Exception as exc:  # nombre duplicado u otra restriccion de la BD
                st.error(tr(f"No se pudo guardar: {exc}", f"Could not save: {exc}"))
            else:
                st.session_state["_saved_scenario"] = {"id": new_id, "name": name.strip()}
                st.session_state.pop("_scenario_docx", None)
                st.success(tr(
                    f"Escenario «{name.strip()}» guardado (id {new_id}).",
                    f"Scenario «{name.strip()}» saved (id {new_id}).",
                ))

    saved = st.session_state.get("_saved_scenario")
    if not saved:
        return

    from views.modelos import build_report

    if st.button(tr(
        f"Generar informe tecnico con el escenario «{saved['name']}»",
        f"Generate technical report with scenario «{saved['name']}»",
    )):
        try:
            with st.spinner(tr("Generando Word…", "Generating Word…")):
                st.session_state["_scenario_docx"] = build_report(
                    "docx", site["id"], lang(), scenario_id=saved["id"], sim_date=on_date
                )
        except DomainError as exc:
            st.session_state.pop("_scenario_docx", None)
            st.error(str(exc))
    content = st.session_state.get("_scenario_docx")
    if content:
        st.download_button(
            tr("Descargar informe tecnico (.docx)", "Download technical report (.docx)"),
            content,
            file_name=(
                f"peattwin_{tr('informe_tecnico_escenario', 'technical_report_scenario')}"
                f"_{saved['id']}.docx"
            ),
            mime=_DOCX_MIME,
            type="primary",
        )
