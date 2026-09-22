"""Carga y preparacion de datos (esqueleto)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.config import RANDOM_SEED


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un dataset procesado desde CSV o Parquet."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def make_synthetic_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Genera un dataset sintetico para validar el pipeline de entrenamiento.

    Variables ficticias asociadas al balance de carbono en turberas:
      - water_table_depth_cm : profundidad del nivel freatico
      - soil_temp_c          : temperatura del suelo
      - precip_mm            : precipitacion
    Objetivo:
      - net_co2_flux         : flujo neto de carbono (ficticio)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    water_table = rng.normal(-15.0, 8.0, n_samples)
    soil_temp = rng.normal(12.0, 4.0, n_samples)
    precip = rng.gamma(2.0, 3.0, n_samples)
    noise = rng.normal(0.0, 0.5, n_samples)

    net_co2_flux = 0.08 * water_table + 0.15 * soil_temp - 0.03 * precip + noise

    return pd.DataFrame(
        {
            "water_table_depth_cm": water_table,
            "soil_temp_c": soil_temp,
            "precip_mm": precip,
            "net_co2_flux": net_co2_flux,
        }
    )
