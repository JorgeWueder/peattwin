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
        m1, m2, m3 = st.columns(3)
        m1.metric("CO₂ · NEE (gC m⁻² d⁻¹)", F.fmt_signed(pred["co2_predicted"], 2))
        m2.metric("CH₄ · FCH4 (nmol m⁻² s⁻¹)", F.fmt(pred["ch4_predicted"], 1))
        m3.metric("Balance de carbono", pred["predicted_class"].capitalize())
        st.caption(
            f"P(sumidero) = **{F.fmt(pred['proba_sumidero'], 3)}** · por signo del NEE: "
            f"**{pred['class_from_nee_sign']}**"
        )
        with st.expander("Supuestos de la simulacion"):
            for a in pred["assumptions"]:
                st.markdown(f"- {a}")

    st.divider()
    st.subheader("Curva de respuesta al nivel freatico")
    st.caption(
        f"{CURVE_POINTS} simulaciones entre {WTD_MIN} y +{WTD_MAX} cm para la fecha "
        "seleccionada. La linea vertical marca el valor actual."
    )
    try:
        curve = _curve(on_date, f"{pred['model_name']}|{pred['model_version']}")
    except DomainError as exc:
        st.error(str(exc))
    else:
        c1, c2 = st.columns(2)
        c1.altair_chart(
            response_curve_chart(curve, metric="co2", marker=wtd, unit="gC m⁻² d⁻¹"),
            width="stretch",
        )
        c2.altair_chart(
            response_curve_chart(curve, metric="ch4", marker=wtd, unit="nmol m⁻² s⁻¹"),
            width="stretch",
        )

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
