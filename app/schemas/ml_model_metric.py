"""Esquemas Pydantic para las metricas de evaluacion de modelos."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MLModelMetricBase(BaseModel):
    # protected_namespaces=() -> permite el campo model_id sin colisionar con "model_"
    model_config = ConfigDict(protected_namespaces=())

    fold: int  # 0..k-1 ; usar -1 para el resultado agregado

    # Regresion
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    nse: float | None = None
    kge: float | None = None

    # Clasificacion
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc_roc: float | None = None

    training_time_seconds: float | None = None


class MLModelMetricCreate(MLModelMetricBase):
    model_id: int


class MLModelMetricRead(MLModelMetricBase):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    model_id: int
    created_at: datetime
