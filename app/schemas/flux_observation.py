"""Esquemas Pydantic para las observaciones de flujo (serie temporal)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from app.models.enums import CarbonBalanceClass, classify_carbon_balance


class FluxObservationBase(BaseModel):
    timestamp: datetime
    nee_co2: float | None = None          # Net Ecosystem Exchange de CO2
    fch4: float | None = None             # flujo de CH4
    water_table_depth: float | None = None  # cm (negativo = bajo la superficie)
    soil_temp: float | None = None       # C
    soil_water_content: float | None = None  # m3/m3
    air_temp: float | None = None        # C
    precipitation: float | None = None   # mm
    carbon_balance_class: CarbonBalanceClass | None = None
    quality_flag: int | None = None

    @model_validator(mode="after")
    def _derive_carbon_balance_class(self) -> "FluxObservationBase":
        """Deriva la clase de balance de carbono del signo del NEE si no viene dada."""
        if self.carbon_balance_class is None and self.nee_co2 is not None:
            self.carbon_balance_class = classify_carbon_balance(self.nee_co2)
        return self


class FluxObservationCreate(FluxObservationBase):
    site_id: int


class FluxObservationUpdate(BaseModel):
    nee_co2: float | None = None
    fch4: float | None = None
    water_table_depth: float | None = None
    soil_temp: float | None = None
    soil_water_content: float | None = None
    air_temp: float | None = None
    precipitation: float | None = None
    carbon_balance_class: CarbonBalanceClass | None = None
    quality_flag: int | None = None


class FluxObservationRead(FluxObservationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    site_id: int
    created_at: datetime
