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
    roles = ", ".join(user.role_names) if user and user.role_names else "sin rol"
    st.warning(
        f"**Vista de solo lectura.** Tu rol ({roles}) no tiene el permiso "
        "`predictions:run`, asi que no puede ejecutar simulaciones. Un usuario con rol "
        "**investigador** o **admin** si puede usar este panel. El resto del dashboard "
        "(series y comparacion de modelos) esta disponible para tu rol."
    )


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


def _explicabilidad_prediccion(wtd: int, on_date) -> str:
    return (
        f"Salida del modelo activo para un unico escenario: nivel freatico llevado a "
        f"{F.fmt_signed(wtd, 0)} cm el {on_date.isoformat()}. NEE en gC m⁻² d⁻¹ con el "
        "signo negativo indicando captura; FCH4 en nmol m⁻² s⁻¹, siempre positivo; y la "
        "clase de balance que devuelve el clasificador. El resto de drivers "
        "—meteorologia y temperatura de suelo— se toman de la climatologia de ese dia "
        "del ano, asi que el escenario es hipotetico y no corresponde a ningun dia real."
    )


def _explicabilidad_curva(gas: str, unidad: str) -> str:
    return (
        f"Respuesta del {gas} al nivel freatico: {CURVE_POINTS} simulaciones "
        f"independientes entre {WTD_MIN} y +{WTD_MAX} cm para la MISMA fecha, con todos "
        f"los demas drivers congelados en su climatologia. El eje X es el nivel freatico "
        f"objetivo en cm (positivo = lamina de agua sobre la superficie) y el Y, el flujo "
        f"predicho en {unidad}. La linea vertical discontinua marca el valor "
        "seleccionado ahora mismo en el control de la izquierda."
    )


def _lectura_prediccion(pred: dict, wtd: int, on_date: date) -> str:
    nee = pred["co2_predicted"]
    ch4 = pred["ch4_predicted"]
    signo = pred["class_from_nee_sign"]
    coincide = signo == pred["predicted_class"]
    total = _co2eq(nee, ch4)
    parte_ch4 = ch4 * _NMOL_A_GCH4 * _GWP100_CH4
    peso = parte_ch4 / abs(total) if total else 0.0
    concordancia = (
        "El clasificador y el signo del NEE coinciden, asi que la etiqueta es solida."
        if coincide
        else (
            "Atencion: el clasificador dice "
            f"**{pred['predicted_class']}** pero el signo del NEE predicho dice "
            f"**{signo}**. Son dos rutas distintas (una cabeza de clasificacion entrenada "
            "con la clase observada frente al signo de la regresion) y su desacuerdo "
            "marca un punto de operacion ambiguo: conviene no apoyar una decision en el."
        )
    )
    return (
        f"Escenario: nivel freatico a {F.fmt_signed(wtd, 0)} cm el {on_date.isoformat()}, con "
        "el resto de drivers tomados de la climatologia de ese dia del anio. "
        f"P(sumidero) = {F.fmt(pred['proba_sumidero'], 3)}. {concordancia} "
        f"Sumando los dos gases en una misma unidad, el balance es de "
        f"{F.fmt_signed(total, 2)} g CO₂-eq m⁻² d⁻¹ (GWP100 = 27), y el metano aporta "
        f"{F.fmt_pct(abs(peso))} de esa magnitud: la etiqueta «sumidero», que mira solo al "
        "NEE, no equivale por si sola a un beneficio climatico neto."
    )


def _forma(serie: pd.Series) -> str:
    """Describe la forma de la curva sin presuponerla.

    La respuesta al nivel freatico NO es monotona en este sitio: decirlo de
    antemano seria afirmar algo que la figura no muestra.
    """
    if serie.is_monotonic_increasing:
        return "crece de forma sostenida"
    if serie.is_monotonic_decreasing:
        return "decrece de forma sostenida"
    i = int(serie.reset_index(drop=True).idxmax())
    j = int(serie.reset_index(drop=True).idxmin())
    ultimo = len(serie) - 1
    if 0 < i < ultimo:
        return "tiene un maximo interior"
    if 0 < j < ultimo:
        return "tiene un minimo interior"
    return "no sigue una direccion unica"


def _lectura_curva_co2(curve: pd.DataFrame, wtd: int) -> str:
    co2 = curve["co2"]
    lo, hi = float(co2.min()), float(co2.max())
    wtd_opt = float(curve.loc[co2.idxmin(), "wtd"])
    rango = hi - lo
    negativos = curve[curve["co2"] < 0]
    positivos = curve[curve["co2"] >= 0]

    if positivos.empty:
        signo = (
            "En TODO el rango simulado el NEE predicho es negativo: para esta fecha el "
            "modelo da captura de CO₂ con cualquier nivel freatico, asi que el manejo del "
            "agua no decide aqui el signo del balance, solo su magnitud"
        )
    elif negativos.empty:
        signo = (
            "En todo el rango simulado el NEE se mantiene positivo: para esta fecha el "
            "modelo no encuentra ningun nivel freatico que convierta el sitio en sumidero"
        )
    else:
        signo = (
            "El NEE cambia de signo dentro del rango: cruza a captura a partir de "
            f"{F.fmt_signed(float(negativos['wtd'].min()), 0)} cm"
        )

    borde = float(wtd_opt) in (float(curve["wtd"].min()), float(curve["wtd"].max()))
    matiz_opt = (
        " Ese minimo cae en el borde del rango simulado, asi que es un limite de la "
        "simulacion y no un optimo: habria que ampliar el rango para saber donde esta."
        if borde
        else ""
    )
    return (
        f"{signo}. La curva {_forma(co2)} y su captura maxima esta en "
        f"{F.fmt_signed(wtd_opt, 0)} cm con {F.fmt_signed(lo, 2)} gC m⁻² d⁻¹.{matiz_opt} "
        f"De un extremo a otro el NEE solo se mueve {F.fmt(rango, 2)} gC m⁻² d⁻¹: ese es "
        "todo el margen que el nivel freatico puede mover en el termino de CO₂ para esta "
        "fecha, y conviene compararlo con el error del modelo en NEE antes de darle "
        "significado —con un R² ≈ 0.2 en esta variable, parte de esa variacion puede ser "
        "ruido del modelo y no respuesta del ecosistema. Ademas la curva mueve UNA variable "
        "con el resto congelado en su climatologia: es sensibilidad del modelo, no una "
        "prediccion de lo que ocurriria al rehumedecer de verdad."
    )


def _lectura_curva_ch4(curve: pd.DataFrame, wtd: int) -> str:
    ch4 = curve["ch4"]
    lo, hi = float(ch4.min()), float(ch4.max())
    wtd_max = float(curve.loc[ch4.idxmax(), "wtd"])
    actual = float(curve.iloc[(curve["wtd"] - wtd).abs().idxmin()]["ch4"])
    variacion = (hi - lo) / lo if lo > 0 else float("inf")
    amplitud = (
        f"la emision varia un {F.fmt_pct(variacion)} entre el minimo y el maximo "
        f"({F.fmt(lo, 1)} → {F.fmt(hi, 1)} nmol m⁻² s⁻¹)"
        if variacion != float("inf")
        else f"la emision va de ≈ 0 a {F.fmt(hi, 1)} nmol m⁻² s⁻¹"
    )
    contexto = (
        " Es una respuesta modesta comparada con el nivel de base: la turbera emite mucho "
        "metano en esta fecha lo haga el nivel freatico lo que haga, y moverlo reordena "
        "poco ese total."
        if variacion < 0.5
        else " La respuesta es grande en terminos relativos, asi que el nivel freatico si "
        "es una palanca real sobre el metano en esta fecha."
    )
    return (
        f"La curva {_forma(ch4)}, con el maximo en {F.fmt_signed(wtd_max, 0)} cm, y "
        f"{amplitud}. En el punto elegido el modelo predice {F.fmt(actual, 1)}.{contexto} "
        "Es el termino del compromiso que se suele omitir cuando se presenta el "
        "rehumedecimiento como beneficio inmediato, y es ademas el que este gemelo estima "
        "con mas fiabilidad (R² ≈ 0.84 en el holdout), porque sus dos drivers "
        "—temperatura de suelo y nivel freatico— estan medidos en el sitio."
    )


def _lectura_compromiso(curve: pd.DataFrame, wtd: int) -> str:
    """Lectura conjunta de las dos curvas: donde esta el optimo en CO2-eq."""
    eq = curve.apply(lambda r: _co2eq(r["co2"], r["ch4"]), axis=1)
    mejor = curve.loc[eq.idxmin()]
    actual_idx = (curve["wtd"] - wtd).abs().idxmin()
    delta = float(eq[actual_idx] - eq.min())
    extremo = float(mejor["wtd"]) in (float(curve["wtd"].min()), float(curve["wtd"].max()))
    borde = (
        " El optimo cae en el borde del rango simulado, asi que es un limite de la "
        "simulacion y no necesariamente un optimo real: habria que ampliar el rango para "
        "saberlo."
        if extremo
        else ""
    )
    situacion = (
        "el punto elegido ya esta en ese minimo"
        if delta < 0.005
        else f"el punto elegido queda {F.fmt(delta, 2)} g CO₂-eq m⁻² d⁻¹ por encima"
    )
    return (
        f"**Lectura conjunta de las dos curvas.** Sumando ambos gases con GWP100 = 27, el "
        f"minimo de emision equivalente de esta fecha esta en {F.fmt_signed(float(mejor['wtd']), 0)} cm "
        f"({F.fmt_signed(float(eq.min()), 2)} g CO₂-eq m⁻² d⁻¹); {situacion}.{borde} "
        "El horizonte temporal "
        "cambia la conclusion: con GWP20 (≈ 80) el peso del metano se triplica y el optimo "
        "se desplaza hacia niveles mas bajos. Por eso la herramienta muestra los dos flujos "
        "por separado y deja la ponderacion explicita en manos de quien decide."
    )


def render() -> None:
    st.title("Simulador de escenarios de restauracion")
    st.caption(
        "Mueve el nivel freatico objetivo y observa la respuesta del ecosistema: NEE "
        "(CO₂), FCH4 (CH₄) y la clase sumidero/fuente que predice el modelo activo. La "
        "meteo y la temperatura del suelo se toman de la climatologia del dia del anio."
    )

    if not auth.has_permission("predictions:run"):
        _read_only_notice()
        return

    controles, resultados = st.columns([1, 2], gap="large")

    st.session_state.setdefault("sim_wtd", 20)

    with controles:
        st.subheader("Parametros del escenario")
        saved = _scenarios()
        if saved:
            options = [None, *saved]
            chosen = st.selectbox(
                "Cargar un escenario guardado",
                options,
                format_func=lambda s: "— ninguno —"
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
            "Nivel freatico objetivo (cm)",
            min_value=WTD_MIN,
            max_value=WTD_MAX,
            step=1,
            key="sim_wtd",
        )
        st.caption(
            "−20 = drenado · 0 = superficie · +80 = inundado. Positivo significa lamina "
            "de agua sobre el suelo: Zarnekow es un fen rehumedecido y opera casi "
            "siempre con valores positivos."
        )
        on_date = st.date_input(
            "Fecha (define la estacionalidad)", DEFAULT_DATE, format="YYYY-MM-DD"
        )

    try:
        pred = _predict(float(wtd), on_date)
    except DomainError as exc:
        with resultados:
            st.error(str(exc))
        return

    with resultados:
        st.subheader("Prediccion para este escenario")
        st.caption(
            f"Modelo activo: **{pred['model_name']} · {pred['model_version']}** "
            f"(formato {pred['model_format']})"
        )
        I.metricas(
            [
                ("CO₂ · NEE (gC m⁻² d⁻¹)", F.fmt_signed(pred["co2_predicted"], 2)),
                ("CH₄ · FCH4 (nmol m⁻² s⁻¹)", F.fmt(pred["ch4_predicted"], 1)),
                ("Balance de carbono", pred["predicted_class"].capitalize()),
            ],
            explicabilidad=_explicabilidad_prediccion(wtd, on_date),
            interpretacion=_lectura_prediccion(pred, wtd, on_date),
        )
        with st.expander("Supuestos de la simulacion"):
            for a in pred["assumptions"]:
                st.markdown(f"- {a}")

    st.divider()
    I.titulo(
        "Curva de respuesta al nivel freatico",
        ayuda=(
            f"{CURVE_POINTS} simulaciones entre {WTD_MIN} y +{WTD_MAX} cm para la fecha "
            "seleccionada. La linea vertical marca el valor actual."
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
    st.subheader("Guardar como escenario")
    st.caption(
        f"Se guarda para el sitio {site['name']} con el nivel freatico actual "
        f"({F.fmt_signed(wtd, 0)} cm)."
    )

    with st.form("save_scenario"):
        name = st.text_input(
            "Nombre", placeholder=f"Nivel freatico a {F.fmt_signed(wtd, 0)} cm"
        )
        description = st.text_input("Descripcion", placeholder="Opcional")
        submitted = st.form_submit_button("Guardar escenario")
    if submitted:
        if not name.strip():
            st.error("El escenario necesita un nombre.")
        else:
            try:
                new_id = _save_scenario(
                    site["id"], name.strip(), float(wtd), description, user.id
                )
            except Exception as exc:  # nombre duplicado u otra restriccion de la BD
                st.error(f"No se pudo guardar: {exc}")
            else:
                st.session_state["_saved_scenario"] = {"id": new_id, "name": name.strip()}
                st.session_state.pop("_scenario_docx", None)
                st.success(f"Escenario «{name.strip()}» guardado (id {new_id}).")

    saved = st.session_state.get("_saved_scenario")
    if not saved:
        return

    from views.modelos import build_report

    if st.button(f"Generar informe tecnico con el escenario «{saved['name']}»"):
        try:
            with st.spinner("Generando Word…"):
                st.session_state["_scenario_docx"] = build_report(
                    "docx", site["id"], scenario_id=saved["id"], sim_date=on_date
                )
        except DomainError as exc:
            st.session_state.pop("_scenario_docx", None)
            st.error(str(exc))
    content = st.session_state.get("_scenario_docx")
    if content:
        st.download_button(
            "Descargar informe tecnico (.docx)",
            content,
            file_name=f"peattwin_informe_tecnico_escenario_{saved['id']}.docx",
            mime=_DOCX_MIME,
            type="primary",
        )
