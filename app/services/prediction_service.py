"""Simulador: escenario -> features -> modelo activo -> prediccion."""
from __future__ import annotations

from datetime import datetime, time, timezone

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.errors import DomainError
from app.ml import predictor
from app.models.prediction import Prediction
from app.models.restoration_scenario import RestorationScenario
from app.schemas.predict import PredictRequest, PredictResponse

# El dataset entreno con WTD en METROS; la columna de BD y la UI usan cm.
_CM_PER_M = 100.0


def run(db: Session, payload: PredictRequest, *, user_id: int | None = None) -> PredictResponse:
    scenario: RestorationScenario | None = None
    site_id: int | None = None

    if payload.scenario_id is not None:
        scenario = db.get(RestorationScenario, payload.scenario_id)
        if scenario is None:
            raise DomainError("Escenario no encontrado", "not_found")
        if scenario.target_water_table_depth is None:
            raise DomainError(
                "El escenario no define `target_water_table_depth`.", "invalid"
            )
        wtd_cm = float(scenario.target_water_table_depth)
        site_id = scenario.site_id
    else:
        wtd_cm = float(payload.target_water_table_depth_cm)

    try:
        active = predictor.load_active_model(db)
        clim, feats = predictor.get_climatology()
    except FileNotFoundError as exc:
        raise DomainError(str(exc), "unavailable")

    row = predictor.build_feature_row(
        clim, feats, wtd_meters=wtd_cm / _CM_PER_M, on_date=payload.on_date
    )
    co2, ch4, proba = active.predict_row(row)

    predicted_class = "sumidero" if proba >= 0.5 else "fuente"
    class_from_sign = "sumidero" if co2 < 0 else "fuente"

    persisted_id: int | None = None
    if payload.persist and scenario is not None and active.regression_model_id is not None:
        persisted_id = _persist(
            db,
            scenario_id=scenario.id,
            model_id=active.classification_model_id or active.regression_model_id,
            on_date=payload.on_date,
            co2=co2,
            ch4=ch4,
            predicted_class="SINK" if predicted_class == "sumidero" else "SOURCE",
        )

    assumptions = [
        f"Nivel freatico mantenido constante a {wtd_cm:.1f} cm ({wtd_cm / _CM_PER_M:.3f} m) "
        "en WTD y sus lags/medias moviles.",
        "Meteo y temperatura de suelo (TS_promedio y derivadas) tomadas de la "
        f"climatologia suavizada del dia del anio {payload.on_date.timetuple().tm_yday} "
        "(media 2016-2018, ventana circular 15 d).",
        "Variables estacionales (doy, month, sin/cos) calculadas de la fecha pedida.",
        f"Modelo activo cargado desde disco sin reentrenar ({active.kind}).",
        "El dataset FLUXNET-CH4 de DE-Zrk no tiene indices de vegetacion (NDVI/LAI); "
        "la incertidumbre del NEE es alta (R2 holdout ~ 0.24).",
    ]

    return PredictResponse(
        co2_predicted=co2,
        ch4_predicted=ch4,
        predicted_class=predicted_class,
        proba_sumidero=proba,
        class_from_nee_sign=class_from_sign,
        model_name=active.name,
        model_version=active.version,
        model_format=active.kind,
        scenario_id=(scenario.id if scenario else None),
        site_id=site_id,
        on_date=payload.on_date,
        target_water_table_depth_cm=wtd_cm,
        persisted_prediction_id=persisted_id,
        assumptions=assumptions,
    )


def _persist(
    db: Session, *, scenario_id: int, model_id: int, on_date, co2: float, ch4: float,
    predicted_class: str,
) -> int:
    ts = datetime.combine(on_date, time(12, 0), tzinfo=timezone.utc)
    stmt = pg_insert(Prediction.__table__).values(
        scenario_id=scenario_id,
        model_id=model_id,
        timestamp=ts,
        co2_predicted=co2,
        ch4_predicted=ch4,
        predicted_class=predicted_class,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_prediction_scenario_model_ts",
        set_={
            "co2_predicted": co2,
            "ch4_predicted": ch4,
            "predicted_class": predicted_class,
        },
    ).returning(Prediction.id)
    new_id = db.execute(stmt).scalar_one()
    db.commit()
    return int(new_id)
