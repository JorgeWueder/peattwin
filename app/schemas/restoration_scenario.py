"""Esquemas Pydantic para los escenarios de restauracion hidrologica."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RestorationScenarioBase(BaseModel):
    name: str
    target_water_table_depth: float | None = None  # cm objetivo
    description: str | None = None


class RestorationScenarioCreate(RestorationScenarioBase):
    site_id: int
    created_by_id: int | None = None


class RestorationScenarioUpdate(BaseModel):
    name: str | None = None
    target_water_table_depth: float | None = None
    description: str | None = None


class RestorationScenarioRead(RestorationScenarioBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    site_id: int
    created_by_id: int | None = None
    created_at: datetime
    updated_at: datetime
