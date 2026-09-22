"""Esquemas para GET /models/comparison (tabla de metricas de todos los modelos)."""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class HoldoutMetrics(BaseModel):
    """Metricas del holdout (fila fold = -1 de ml_model_metrics)."""

    model_config = ConfigDict(protected_namespaces=())

    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    nse: float | None = None
    kge: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc_roc: float | None = None
    training_time_seconds: float | None = None


class CvSummary(BaseModel):
    """Agregado de los folds 0..k-1 de la validacion cruzada."""

    n_folds: int
    rmse_mean: float | None = None
    rmse_std: float | None = None
    r2_mean: float | None = None
    r2_std: float | None = None
    f1_mean: float | None = None
    f1_std: float | None = None
    auc_mean: float | None = None
    auc_std: float | None = None


class ModelComparisonRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)

    model_id: int
    name: str
    task: str                     # REGRESSION | CLASSIFICATION
    algorithm_type: str           # CLASSICAL | HYBRID
    framework: str | None = None
    version: str
    is_active: bool
    target_variable: str | None = None
    trained_at: datetime | None = None
    holdout: HoldoutMetrics | None = None
    cv: CvSummary | None = None


class ModelComparisonResponse(BaseModel):
    generated_at: datetime
    n_models: int
    criterio_seleccion: str = Field(
        default=(
            "combined = 0.5 * reg_score_norm + 0.5 * (0.5 * F1_macro + 0.5 * AUC); "
            "modelo desplegado: version con is_active = true"
        )
    )
    rows: list[ModelComparisonRow]


class PredictedSeriesPoint(BaseModel):
    """Un dia: valor observado vs predicho por el modelo activo (backtest)."""

    date: date
    nee_observed: float | None = None
    nee_predicted: float | None = None
    fch4_observed: float | None = None
    fch4_predicted: float | None = None
    class_observed: str | None = None
    class_predicted: str | None = None


class PredictedSeriesResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    site_id: int
    model_name: str
    model_version: str
    n_points: int
    points: list[PredictedSeriesPoint]
