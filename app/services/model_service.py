"""Servicio de modelos de ML: listado, tabla comparativa y activacion."""
from __future__ import annotations

import statistics
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import DomainError
from app.models.ml_model import MLModel
from app.models.ml_model_metric import MLModelMetric
from app.models.site import Site
from app.schemas.model_comparison import (
    CvSummary,
    HoldoutMetrics,
    ModelComparisonResponse,
    ModelComparisonRow,
    PredictedSeriesPoint,
    PredictedSeriesResponse,
)

_HOLDOUT_FOLD = -1
_MAX_SERIES_POINTS = 1500


def list_models(db: Session) -> list[MLModel]:
    return list(db.scalars(select(MLModel).order_by(MLModel.id)))


def active_models(db: Session) -> list[MLModel]:
    return list(
        db.scalars(select(MLModel).where(MLModel.is_active.is_(True)).order_by(MLModel.task))
    )


def _agg(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def comparison(db: Session) -> ModelComparisonResponse:
    models = list_models(db)
    metrics = list(db.scalars(select(MLModelMetric)))
    by_model: dict[int, list[MLModelMetric]] = {}
    for m in metrics:
        by_model.setdefault(m.model_id, []).append(m)

    rows: list[ModelComparisonRow] = []
    for model in models:
        mm = by_model.get(model.id, [])
        holdout = next((x for x in mm if x.fold == _HOLDOUT_FOLD), None)
        cv_folds = [x for x in mm if x.fold >= 0]

        h = None
        if holdout is not None:
            h = HoldoutMetrics(
                rmse=holdout.rmse, mae=holdout.mae, r2=holdout.r2,
                nse=holdout.nse, kge=holdout.kge,
                accuracy=holdout.accuracy, precision=holdout.precision,
                recall=holdout.recall, f1=holdout.f1, auc_roc=holdout.auc_roc,
                training_time_seconds=holdout.training_time_seconds,
            )

        cv = None
        if cv_folds:
            rmse_m, rmse_s = _agg([x.rmse for x in cv_folds])
            r2_m, r2_s = _agg([x.r2 for x in cv_folds])
            f1_m, f1_s = _agg([x.f1 for x in cv_folds])
            auc_m, auc_s = _agg([x.auc_roc for x in cv_folds])
            cv = CvSummary(
                n_folds=len({x.fold for x in cv_folds}),
                rmse_mean=rmse_m, rmse_std=rmse_s,
                r2_mean=r2_m, r2_std=r2_s,
                f1_mean=f1_m, f1_std=f1_s,
                auc_mean=auc_m, auc_std=auc_s,
            )

        rows.append(
            ModelComparisonRow(
                model_id=model.id,
                name=model.name,
                task=str(model.task.value if hasattr(model.task, "value") else model.task),
                algorithm_type=str(
                    model.algorithm_type.value
                    if hasattr(model.algorithm_type, "value")
                    else model.algorithm_type
                ),
                framework=model.framework,
                version=model.version,
                is_active=model.is_active,
                target_variable=model.target_variable,
                trained_at=model.trained_at,
                holdout=h,
                cv=cv,
            )
        )

    return ModelComparisonResponse(
        generated_at=datetime.now(timezone.utc),
        n_models=len(rows),
        rows=rows,
    )


def activate(db: Session, model_id: int) -> MLModel | None:
    """Marca un modelo como activo y desactiva los demas de la misma tarea."""
    model = db.get(MLModel, model_id)
    if model is None:
        return None
    for other in db.scalars(
        select(MLModel).where(MLModel.task == model.task, MLModel.is_active.is_(True))
    ):
        other.is_active = False
    model.is_active = True
    db.commit()
    db.refresh(model)
    return model


def predicted_series(
    db: Session, *, site_id: int, start: date | None = None, end: date | None = None
) -> PredictedSeriesResponse:
    """Backtest: observado vs predicho por el modelo activo, dia a dia.

    Solo disponible para el sitio con dataset procesado (DE-Zrk). Usa las features
    REALES de ese dia (no la climatologia del simulador).
    """
    from app.ml import predictor

    site = db.get(Site, site_id)
    if site is None:
        raise DomainError("Sitio no encontrado", "not_found")
    if site.fluxnet_id != "DE-Zrk":
        raise DomainError(
            "No hay dataset procesado para este sitio (solo DE-Zrk).", "not_found"
        )
    try:
        active = predictor.load_active_model(db)
        df, feats = predictor.feature_frame()
    except FileNotFoundError as exc:
        raise DomainError(str(exc), "unavailable")

    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.sort_values("date")
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    df = df.tail(_MAX_SERIES_POINTS)

    points: list[PredictedSeriesPoint] = []
    if not df.empty:
        nee_p, fch4_p, proba = predictor.predict_frame(active, df[feats])
        nee_obs = df["NEE_F_ANNOPTLM"].tolist()
        fch4_obs = df["FCH4_F_ANNOPTLM"].tolist()
        dates = df["date"].tolist()
        for i in range(len(df)):
            points.append(
                PredictedSeriesPoint(
                    date=dates[i],
                    nee_observed=float(nee_obs[i]),
                    nee_predicted=nee_p[i],
                    fch4_observed=float(fch4_obs[i]),
                    fch4_predicted=fch4_p[i],
                    class_observed="sumidero" if nee_obs[i] < 0 else "fuente",
                    class_predicted="sumidero" if proba[i] >= 0.5 else "fuente",
                )
            )

    return PredictedSeriesResponse(
        site_id=site_id,
        model_name=active.name,
        model_version=active.version,
        n_points=len(points),
        points=points,
    )
