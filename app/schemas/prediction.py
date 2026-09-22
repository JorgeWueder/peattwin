"""Esquemas Pydantic para las predicciones del modelo activo."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import CarbonBalanceClass


class PredictionBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    timestamp: datetime
    co2_predicted: float | None = None
    ch4_predicted: float | None = None
    predicted_class: CarbonBalanceClass | None = None


class PredictionCreate(PredictionBase):
    scenario_id: int
    model_id: int


class PredictionRead(PredictionBase):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    scenario_id: int
    model_id: int
    created_at: datetime
