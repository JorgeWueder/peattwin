"""Parametros del ETL de DE-Zrk.

Todo lo configurable del pipeline vive aqui para que el informe pueda citar
un unico sitio de verdad (single source of truth).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FLX_DE-Zrk_FLUXNET-CH4_DD_2013-2018_1-1.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUT_OBSERVATIONS = PROCESSED_DIR / "DE-Zrk_flux_daily_2016-2018.parquet"
OUT_ML_DATASET = PROCESSED_DIR / "DE-Zrk_ml_dataset_2016-2018.parquet"
OUT_METADATA = PROCESSED_DIR / "DE-Zrk_etl_metadata.json"

ENV_FILE = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------- metadatos del sitio
# Fuente: FLX_AA-Flx_CH4-META (FLUXNET-CH4 2020). Zarnekow, Alemania.
# Fen (turbera minerotrofica) inundada/rehumedecida hidrologicamente en 2004-2005.
SITE = {
    "fluxnet_id": "DE-Zrk",
    "name": "Zarnekow",
    "country": "Germany",
    "latitude": 53.87594,
    "longitude": 12.88901,
    "peatland_type": "FEN",
    "utc_offset_hours": 1,  # hora estandar local del sitio (UTC+1, sin DST en agregados diarios)
    "koppen": "Dfb",
    "igbp": "WET",
}

# ------------------------------------------------------------------ constantes ETL
MISSING_VALUE = -9999          # centinela de dato faltante de FLUXNET
TIMESTAMP_FORMAT = "%Y%m%d"    # TIMESTAMP viene como entero YYYYMMDD

# Periodo de analisis: SOLO 2016-2018 (ver justificacion en run_etl.py, paso 4)
ANALYSIS_YEARS = (2016, 2017, 2018)
TRAIN_YEARS = (2016, 2017)     # split temporal (NO aleatorio)
TEST_YEARS = (2018,)

# ------------------------------------------------------------------------ objetivos
# Regresion: variables ya rellenadas al 100 % por el equipo del sitio con red
# neuronal (ANNOPTLM). Se usan tal cual -> no implementamos gap-filling propio.
TARGET_NEE = "NEE_F_ANNOPTLM"
TARGET_FCH4 = "FCH4_F_ANNOPTLM"
TARGETS_REGRESSION = [TARGET_NEE, TARGET_FCH4]

# Clasificacion derivada del signo del NEE gap-filled.
TARGET_CLASS = "clase_balance_carbono"
CLASS_SINK = "sumidero"   # NEE_F_ANNOPTLM <  0
CLASS_SOURCE = "fuente"   # NEE_F_ANNOPTLM >= 0

# -------------------------------------------------------------------- features base
# Drivers meteorologicos gap-filled (_F) con cobertura 100 % en 2016-2018
# (verificado en el ETL). Se excluyen PPFD_IN_F (65 dias vacios en 2018) y
# PPFD_OUT_F/SW_OUT_F/LW_OUT_F por redundancia con SW_IN_F/NETRAD_F.
METEO_FEATURES = [
    "TA_F",       # temperatura del aire (C)
    "P_F",        # precipitacion (mm d-1)
    "VPD_F",      # deficit de presion de vapor (hPa)
    "SW_IN_F",    # radiacion de onda corta entrante (W m-2)
    "RH_F",       # humedad relativa (%)
    "WS_F",       # velocidad del viento (m s-1)
    "PA_F",       # presion atmosferica (kPa)
    "NETRAD_F",   # radiacion neta (W m-2)
    "LW_IN_F",    # radiacion de onda larga entrante (W m-2)
    "G_F",        # flujo de calor del suelo (W m-2)
]
# Hidrologia y temperatura del suelo usadas como FEATURES de ML.
# Solo TS_1 y TS_2: TS_3..TS_5 estan muy correlacionadas con TS_1 (r>0.95) y
# aportarian multicolinealidad. El perfil completo si se conserva en el parquet
# de observaciones para el EDA (SOIL_TEMP_PROFILE).
HYDRO_SOIL_FEATURES = [
    "WTD",        # nivel freatico (m; positivo = lamina de agua sobre la superficie)
    "TS_1",       # temperatura del suelo a 0.05 m (C) -> "superficial"
    "TS_2",       # temperatura del suelo a 0.10 m (C)
]

# Perfil termico del suelo por profundidad (FLUXNET-CH4 DE-Zrk):
#   TS_1=-0.05 m  TS_2=-0.10 m  TS_3=-0.20 m  TS_4=-0.32 m  TS_5=-0.48 m aprox.
# Se guarda completo en el parquet de observaciones (no como feature de ML).
SOIL_TEMP_PROFILE = ["TS_1", "TS_2", "TS_3", "TS_4", "TS_5"]

# Columnas de flujo que NUNCA entran como feature (fuga de informacion hacia el
# objetivo): NEE/NEE_F, H/H_F, LE/LE_F, FCH4/FCH4_F, GPP_*, RECO_*, USTAR.
LEAKAGE_COLUMNS = [
    "NEE", "NEE_F", "H", "H_F", "LE", "LE_F", "FCH4", "FCH4_F",
    "GPP_NT", "RECO_NT", "GPP_DT", "RECO_DT", "USTAR", "LE_F_ANNOPTLM",
]

# ----------------------------------------------------- limpieza / gap-filling propio
# Gaps internos cortos en WTD y en el perfil TS_1..TS_5 dentro de 2016-2018
# (14 dias afectados, mismo patron en todas: 1 dia en jun-2016 + tramo de 7 dias
# en abr/may-2017 + 6 dias sueltos; gap_max=7 d). Se rellenan por interpolacion
# lineal temporal con limite de INTERP_LIMIT dias consecutivos.
INTERP_VARS = ["WTD", "TS_1", "TS_2", "TS_3", "TS_4", "TS_5"]
INTERP_LIMIT_DAYS = 7

# --------------------------------------------------------------- features derivadas
LAG_VARS = ["WTD", "TS_1"]     # (paso 5) lags de nivel freatico y temp. suelo superficial
LAGS_DAYS = [1, 3, 7]

ROLL_MEAN_VARS = ["WTD", "TS_1", "TA_F"]   # medias moviles
ROLL_SUM_VARS = ["P_F"]                    # precipitacion -> suma movil (acumulado)
ROLL_WINDOWS_DAYS = [7, 30]

# Mapa parquet -> tabla flux_observations (lo usa load_to_db.py)
DB_COLUMN_MAP = {
    "nee_co2": TARGET_NEE,          # gC m-2 d-1 (negativo = captura)
    "fch4": TARGET_FCH4,            # nmol CH4 m-2 s-1 (media diaria)
    "water_table_depth": "WTD",    # se convierte de m a cm en el loader
    "soil_temp": "TS_1",           # C
    "air_temp": "TA_F",            # C
    "precipitation": "P_F",        # mm d-1
    # soil_water_content: NO disponible en este dataset -> NULL
    "quality_flag": "FCH4_F_ANNOPTLM_QC",
}
WTD_METERS_TO_CM = 100.0
