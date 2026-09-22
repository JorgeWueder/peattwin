"""Esquemas para POST /predict (simulador de escenarios de restauracion)."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scenario_id: int | None = Field(
        default=None,
        description="Si se indica, se usa su `target_water_table_depth` y su `site_id`.",
        examples=[1],
    )
    target_water_table_depth_cm: float | None = Field(
        default=None,
        description="Nivel freatico objetivo en cm (positivo = lamina de agua sobre "
        "la superficie). Obligatorio si no se pasa `scenario_id`.",
        examples=[20.0],
    )
    on_date: date = Field(
        ...,
        alias="date",
        description="Fecha para la que se predice; define la estacionalidad "
        "(climatologia del dia del anio).",
        examples=["2019-07-15"],
    )
    persist: bool = Field(
        default=False,
        description="Si hay `scenario_id`, guarda el resultado en la tabla `predictions`.",
    )

    @model_validator(mode="after")
    def _need_wtd_or_scenario(self) -> "PredictRequest":
        if self.scenario_id is None and self.target_water_table_depth_cm is None:
            raise ValueError(
                "Indica `scenario_id` o `target_water_table_depth_cm`."
            )
        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    co2_predicted: float = Field(description="NEE predicho (gC m-2 d-1). Negativo = sumidero.")
    ch4_predicted: float = Field(description="FCH4 predicho (nmol CH4 m-2 s-1).")
    predicted_class: str = Field(description="'sumidero' o 'fuente' (clasificador del modelo activo).")
    proba_sumidero: float = Field(description="Probabilidad de la clase 'sumidero'.")
    class_from_nee_sign: str = Field(description="Clase derivada del signo de co2_predicted (control).")

    model_name: str
    model_version: str
    model_format: str = Field(description="'pkl' (scikit-learn/stacking) o 'h5' (Keras/CNN-LSTM).")

    scenario_id: int | None = None
    site_id: int | None = None
    on_date: date
    target_water_table_depth_cm: float

    persisted_prediction_id: int | None = None
    assumptions: list[str]
