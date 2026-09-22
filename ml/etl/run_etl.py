"""ETL de flujos de carbono - sitio FLUXNET-CH4 DE-Zrk (Zarnekow, Alemania).

===============================================================================
CONTEXTO
===============================================================================
DE-Zrk / Zarnekow: turbera tipo *fen* (minerotrofica) del NE de Alemania,
rehumedecida hidrologicamente en 2004-2005 (elevacion del nivel freatico por
cierre de drenajes). Producto FLUXNET-CH4 2015, resolucion diaria (DD),
2013-2018, 2191 filas.

Este script cubre los pasos 1-7 del enunciado y produce DOS parquet en
data/processed/ mas un JSON de metadatos con la trazabilidad completa para el
informe:

  1) DE-Zrk_flux_daily_2016-2018.parquet   -> observaciones diarias limpias
     (1096 filas, lo carga load_to_db.py en la tabla flux_observations)
  2) DE-Zrk_ml_dataset_2016-2018.parquet   -> matriz lista para ML: objetivos +
     features derivadas + columna 'split' (train/test temporal)
  3) DE-Zrk_etl_metadata.json              -> fuente + hash, conteos por paso,
     log de limpieza, supuestos, balance de clases

Ejecutar desde la raiz del repositorio:

    python -m ml.etl.run_etl

===============================================================================
SUPUESTOS Y DECISIONES DE LIMPIEZA  (sustento para el informe)
===============================================================================
A. Centinela de faltantes: -9999 (convencion FLUXNET) -> NaN. Verificado que no
   quedan -9999 residuales tras la conversion.
B. TIMESTAMP es un entero YYYYMMDD -> fecha. La serie diaria 2013-2018 es
   CONTINUA: 2191 filas == 2191 dias, 0 huecos de calendario. Esto permite que
   .shift(n) sea exactamente un desfase de n dias y que .rolling(k) sea una
   ventana de k dias naturales.
C. Objetivos de regresion = NEE_F_ANNOPTLM y FCH4_F_ANNOPTLM. Ya vienen
   rellenados al 100 % por el equipo del sitio con un metodo de red neuronal
   (ANNOPTLM). Se usan tal cual: NO se implementa gap-filling propio.
   Unidades (protocolo FLUXNET-CH4): NEE en gC m-2 d-1 (negativo = sumidero);
   FCH4 en nmol CH4 m-2 s-1 (media diaria).
D. Objetivo de clasificacion 'clase_balance_carbono' derivado del signo del NEE
   gap-filled:  NEE_F_ANNOPTLM <  0  -> "sumidero"
                NEE_F_ANNOPTLM >= 0  -> "fuente"
   (definicion del enunciado; no hay clase "neutro"). Es una clasificacion
   desbalanceada: ~30 % sumidero / ~70 % fuente en 2016-2018 (se reporta en el
   JSON). Coherente con un fen rehumedecido: fuerte fuente de CH4 y de CO2 en
   respiracion durante gran parte del anio.
E. PERIODO 2016-2018 (paso 4). Se DESCARTAN 2013, 2014 y 2015 porque WTD
   (nivel freatico) esta 100 % VACIO en esos tres anios -- tambien su version
   gap-filled WTD_F. Cobertura de WTD por anio (verificada por el ETL):
        2013: 0 %   2014: 0 %   2015: 0 %   2016: 100 %   2017: 96 %   2018: 100 %
   El nivel freatico es una variable de entrada OBLIGATORIA: el simulador de
   escenarios de restauracion del dashboard (Paso 10) perturba WTD para generar
   los escenarios "y si el nivel freatico sube/baja X" y predice la respuesta de
   CO2/CH4. Sin WTD no hay eje de escenario y el simulador no puede funcionar.
   Conservar 2013-2015 obligaria a imputar el 100 % del driver de interes, lo
   cual no es defendible metodologicamente. Resultado del filtro: 1096 filas
   (2016-01-01 .. 2018-12-31).
F. Gap-filling propio MINIMO y solo sobre variables de ENTRADA: gaps internos
   cortos de WTD/TS_1/TS_2 dentro de 2016-2018 (14 dias afectados: 1 dia en
   jun-2016, un tramo de 7 dias en abr/may-2017 y 6 dias sueltos) se rellenan
   por interpolacion lineal temporal con limite de 7 dias consecutivos. Se
   marcan las filas afectadas con banderas booleanas (*_gapfilled_by_etl).
G. FEATURES (paso 5). Se excluyen como predictores todas las columnas de flujo
   (NEE*, H*, LE*, FCH4*, GPP_*, RECO_*, USTAR): son el objetivo o derivados del
   objetivo -> fuga de informacion. Se excluye PPFD_IN_F (65 dias vacios en
   ene-mar 2018) y se usa SW_IN_F como proxy de radiacion. Drivers usados:
   meteo gap-filled (_F) con cobertura 100 % + WTD + TS_1 + TS_2.
   Derivadas: lags [1,3,7] d de WTD y TS_1; medias moviles 7/30 d de WTD, TS_1 y
   TA_F; suma movil 7/30 d de P_F (acumulado de precipitacion); estacionales dia
   del anio y mes, mas su codificacion ciclica seno/coseno (evita el salto
   artificial 31-dic -> 1-ene).
H. Las ventanas moviles usan min_periods = ventana completa: las filas sin 30
   dias previos (enero de 2016) quedan con NaN y SE DESCARTAN de la matriz ML
   (se pierden los primeros 29 dias: 1096 -> 1067 filas aprox., cifra exacta en
   el JSON). La tabla de observaciones (parquet 1) conserva las 1096 filas.
I. SPLIT TEMPORAL (paso 6), NUNCA aleatorio: train = 2016-2017, test = 2018.
   Entrenar en los dos primeros anios y validar en el ultimo anio completo imita
   el uso operativo (pronostico hacia adelante) y respeta la autocorrelacion
   temporal.
J. Marco de tiempo: cada registro diario representa un dia natural. En los
   parquet se guarda 'timestamp' como fecha naive a medianoche; load_to_db.py lo
   ancla a las 12:00 UTC al insertar, para que 'timestamp::date' devuelva el dia
   correcto en cualquier zona horaria de sesion. El desfase de hora local del
   sitio (UTC+1) queda solo en los metadatos.
K. soil_water_content NO existe en este dataset -> se deja NULL en
   flux_observations. WTD se convierte de m a cm al cargar en BD (la columna
   water_table_depth esta documentada en cm); en Zarnekow WTD es mayoritariamente
   positivo (lamina de agua sobre el suelo, sitio inundado).
===============================================================================
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.etl import config as C
from ml import _logging


# --------------------------------------------------------------------------- utils
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CleaningLog:
    """Acumula entradas para el JSON de metadatos."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, step: str, description: str, **extra) -> None:
        entry = {"step": step, "description": description}
        entry.update(extra)
        self.steps.append(entry)
        detail = "  ".join(f"{k}={v}" for k, v in extra.items())
        print(f"[{step}] {description}" + (f"  ({detail})" if detail else ""))


# ------------------------------------------------------------------ paso 1: extract
def load_raw(log: CleaningLog) -> pd.DataFrame:
    """Carga el CSV, parsea TIMESTAMP y convierte -9999 -> NaN."""
    # na_values captura el centinela ya en la lectura (int y float).
    df = pd.read_csv(
        C.RAW_CSV,
        na_values=[C.MISSING_VALUE, float(C.MISSING_VALUE), str(C.MISSING_VALUE)],
    )
    n_raw = len(df)

    df["timestamp"] = pd.to_datetime(
        df["TIMESTAMP"].astype("Int64").astype(str), format=C.TIMESTAMP_FORMAT
    )

    # Red de seguridad: cualquier -9999 que se hubiera colado.
    num_cols = df.select_dtypes(include=[np.number]).columns
    residual = int((df[num_cols] == C.MISSING_VALUE).sum().sum())
    df[num_cols] = df[num_cols].replace(C.MISSING_VALUE, np.nan)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Continuidad de la serie diaria.
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="D")
    missing_days = len(full) - df["timestamp"].nunique()

    log.add(
        "1-extract",
        "CSV cargado; TIMESTAMP(YYYYMMDD)->fecha; -9999->NaN",
        source_file=C.RAW_CSV.name,
        rows=n_raw,
        columns=df.shape[1] - 1,
        date_range=[str(df["timestamp"].min().date()), str(df["timestamp"].max().date())],
        residual_sentinels_replaced=residual,
        missing_calendar_days=missing_days,
    )
    if missing_days:
        # La serie deberia ser continua; si no lo fuera, se reindexa a diario.
        df = (
            df.set_index("timestamp")
            .reindex(full)
            .rename_axis("timestamp")
            .reset_index()
        )
        log.add("1b-extract", "Serie reindexada a diario (huecos -> NaN)", added_rows=missing_days)
    return df


# ------------------------------------------------ pasos 2-3: objetivos y etiqueta
def add_targets(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    for col in (C.TARGET_NEE, C.TARGET_FCH4):
        cov = df[col].notna().mean() * 100
        if cov < 100:
            log.add(
                "2-targets",
                f"AVISO: {col} no esta al 100 % ({cov:.1f} %) en el crudo completo",
            )

    # Paso 3: clasificacion derivada del signo del NEE gap-filled.
    sink = df[C.TARGET_NEE] < 0
    df[C.TARGET_CLASS] = np.where(sink, C.CLASS_SINK, C.CLASS_SOURCE)
    df[f"{C.TARGET_CLASS}_cod"] = sink.astype("int8")  # 1 = sumidero, 0 = fuente
    df.loc[df[C.TARGET_NEE].isna(), [C.TARGET_CLASS, f"{C.TARGET_CLASS}_cod"]] = np.nan

    log.add(
        "2-3-targets",
        "Objetivos de regresion (NEE/FCH4 _F_ANNOPTLM) y etiqueta de clasificacion",
        regression_targets=C.TARGETS_REGRESSION,
        classification_rule="NEE_F_ANNOPTLM<0 -> sumidero ; >=0 -> fuente",
    )
    return df


# ------------------------------------------------------- paso 4: filtro 2016-2018
def filter_period(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    year = df["timestamp"].dt.year
    wtd_cov = (
        df.assign(y=year)
        .groupby("y")["WTD"]
        .apply(lambda s: round(float(s.notna().mean()) * 100, 1))
        .to_dict()
    )
    out = df[year.isin(C.ANALYSIS_YEARS)].reset_index(drop=True)
    log.add(
        "4-filter",
        "Filtrado a 2016-2018 (WTD 100% vacio en 2013-2015; driver obligatorio "
        "para el simulador de escenarios del dashboard, Paso 10)",
        wtd_coverage_pct_by_year=wtd_cov,
        rows_in=len(df),
        rows_out=len(out),
        discarded_years=[y for y in sorted(set(year)) if y not in C.ANALYSIS_YEARS],
    )
    return out


# ------------------------------------------- limpieza F: interpolacion de entradas
def interpolate_inputs(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = df.set_index("timestamp")
    affected = {}
    for col in C.INTERP_VARS:
        before = df[col].isna()
        flag = f"{col.lower()}_gapfilled_by_etl"
        df[flag] = False
        df[col] = df[col].interpolate(method="time", limit=C.INTERP_LIMIT_DAYS)
        filled = before & df[col].notna()
        df.loc[filled, flag] = True
        affected[col] = int(filled.sum())
    df = df.reset_index()
    log.add(
        "4b-clean",
        f"Interpolacion lineal temporal de {C.INTERP_VARS} (limite "
        f"{C.INTERP_LIMIT_DAYS} d); filas marcadas con *_gapfilled_by_etl",
        days_filled=affected,
    )
    return df


# ---------------------------------------------------- paso 5: features derivadas
def engineer_features(df: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols: list[str] = list(C.METEO_FEATURES) + list(C.HYDRO_SOIL_FEATURES)

    # Lags (desfase en dias; la serie es diaria y continua).
    for var in C.LAG_VARS:
        for lag in C.LAGS_DAYS:
            name = f"{var}_lag{lag}"
            df[name] = df[var].shift(lag)
            feature_cols.append(name)

    # Medias moviles (ventana completa obligatoria: min_periods == window).
    for var in C.ROLL_MEAN_VARS:
        for w in C.ROLL_WINDOWS_DAYS:
            name = f"{var}_roll{w}_mean"
            df[name] = df[var].rolling(window=w, min_periods=w).mean()
            feature_cols.append(name)

    # Suma movil de precipitacion (acumulado).
    for var in C.ROLL_SUM_VARS:
        for w in C.ROLL_WINDOWS_DAYS:
            name = f"{var}_roll{w}_sum"
            df[name] = df[var].rolling(window=w, min_periods=w).sum()
            feature_cols.append(name)

    # Estacionales + codificacion ciclica.
    doy = df["timestamp"].dt.dayofyear
    month = df["timestamp"].dt.month
    df["doy"] = doy.astype("int16")
    df["month"] = month.astype("int8")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    feature_cols += ["doy", "month", "doy_sin", "doy_cos", "month_sin", "month_cos"]

    log.add(
        "5-features",
        "Lags [1,3,7]d de WTD/TS_1; medias moviles 7/30d de WTD/TS_1/TA_F; "
        "suma movil 7/30d de P_F; estacionales (doy, month) + seno/coseno",
        n_features=len(feature_cols),
    )
    return df, feature_cols


# ---------------------------------------------------- paso 6: split temporal
def temporal_split(df: pd.DataFrame, feature_cols: list[str], log: CleaningLog) -> pd.DataFrame:
    needed = feature_cols + C.TARGETS_REGRESSION + [C.TARGET_CLASS]
    before = len(df)
    ml = df.dropna(subset=needed).reset_index(drop=True)

    year = ml["timestamp"].dt.year
    ml["split"] = np.select(
        [year.isin(C.TRAIN_YEARS), year.isin(C.TEST_YEARS)],
        ["train", "test"],
        default="unused",
    )
    log.add(
        "6-split",
        "Split TEMPORAL (no aleatorio): train=2016-2017, test=2018. "
        f"Descartadas {before - len(ml)} filas sin ventana movil completa / sin objetivo",
        rows_ml=len(ml),
        train_rows=int((ml["split"] == "train").sum()),
        test_rows=int((ml["split"] == "test").sum()),
        ml_date_range=[str(ml["timestamp"].min().date()), str(ml["timestamp"].max().date())],
    )
    return ml


# ---------------------------------------------------- paso 7: escritura de salidas
def _class_balance(s: pd.Series) -> dict:
    vc = s.value_counts(dropna=False)
    total = int(vc.sum())
    return {str(k): {"n": int(v), "pct": round(100 * v / total, 1)} for k, v in vc.items()}


def write_outputs(
    obs: pd.DataFrame,
    ml: pd.DataFrame,
    feature_cols: list[str],
    log: CleaningLog,
) -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -- Parquet 1: observaciones diarias limpias (para flux_observations) --
    obs_cols = (
        [
            "timestamp",
            C.TARGET_NEE,
            C.TARGET_FCH4,
            "FCH4_F_ANNOPTLM_QC",
            C.TARGET_CLASS,
            f"{C.TARGET_CLASS}_cod",
            "WTD",
        ]
        + list(C.SOIL_TEMP_PROFILE)   # TS_1..TS_5 (perfil completo para el EDA)
        + ["TA_F", "P_F", "wtd_gapfilled_by_etl"]
        + [f"{c.lower()}_gapfilled_by_etl" for c in C.SOIL_TEMP_PROFILE]
    )
    obs_out = obs[[c for c in obs_cols if c in obs.columns]].copy()
    obs_out.insert(0, "fluxnet_id", C.SITE["fluxnet_id"])
    obs_out.to_parquet(C.OUT_OBSERVATIONS, index=False, engine="pyarrow")

    # -- Parquet 2: matriz ML --
    ml_keep = (
        ["fluxnet_id", "timestamp", "split"]
        + C.TARGETS_REGRESSION
        + [C.TARGET_CLASS, f"{C.TARGET_CLASS}_cod"]
        + feature_cols
    )
    ml_out = ml.copy()
    ml_out.insert(0, "fluxnet_id", C.SITE["fluxnet_id"])
    ml_out = ml_out[[c for c in ml_keep if c in ml_out.columns]]
    ml_out.to_parquet(C.OUT_ML_DATASET, index=False, engine="pyarrow")

    # -- JSON de metadatos --
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "file": C.RAW_CSV.name,
            "sha256": _sha256(C.RAW_CSV),
            "rows": int(pd.read_csv(C.RAW_CSV, usecols=[0]).shape[0]),
        },
        "site": C.SITE,
        "targets": {
            "regression": C.TARGETS_REGRESSION,
            "regression_units": {
                C.TARGET_NEE: "gC m-2 d-1 (negativo = sumidero)",
                C.TARGET_FCH4: "nmol CH4 m-2 s-1 (media diaria)",
            },
            "classification": {
                "column": C.TARGET_CLASS,
                "rule": "NEE_F_ANNOPTLM < 0 -> 'sumidero' ; >= 0 -> 'fuente'",
            },
        },
        "features": feature_cols,
        "n_features": len(feature_cols),
        "class_balance": {
            "full_2016_2018": _class_balance(obs[C.TARGET_CLASS]),
            "train": _class_balance(ml.loc[ml["split"] == "train", C.TARGET_CLASS]),
            "test": _class_balance(ml.loc[ml["split"] == "test", C.TARGET_CLASS]),
        },
        "split": {
            "type": "temporal (no aleatorio)",
            "train_years": list(C.TRAIN_YEARS),
            "test_years": list(C.TEST_YEARS),
            "train_rows": int((ml["split"] == "train").sum()),
            "test_rows": int((ml["split"] == "test").sum()),
        },
        "row_counts": {
            "raw": int(pd.read_csv(C.RAW_CSV, usecols=[0]).shape[0]),
            "observations_2016_2018": len(obs_out),
            "ml_dataset": len(ml_out),
        },
        "cleaning_log": log.steps,
        "assumptions": [
            "Centinela de faltantes -9999 -> NaN (convencion FLUXNET).",
            "Serie diaria 2013-2018 continua (0 huecos de calendario).",
            "NEE_F_ANNOPTLM / FCH4_F_ANNOPTLM: gap-filled 100% por el sitio "
            "(red neuronal ANNOPTLM); se usan sin gap-filling propio.",
            "2013-2015 descartados: WTD y WTD_F 100% vacios; WTD es driver "
            "obligatorio del simulador de escenarios del dashboard (Paso 10).",
            "Interpolacion lineal temporal (limite 7 d) SOLO en entradas "
            "WTD/TS_1/TS_2; filas marcadas con *_gapfilled_by_etl.",
            "Columnas de flujo (NEE*, H*, LE*, FCH4*, GPP_*, RECO_*, USTAR) "
            "excluidas como features (fuga de informacion).",
            "PPFD_IN_F excluido (65 dias vacios en 2018); SW_IN_F como proxy.",
            "Ventanas moviles con min_periods completo -> se descartan los "
            "primeros 29 dias de 2016 en la matriz ML.",
            "Split temporal train=2016-2017 / test=2018 (nunca aleatorio).",
            "soil_water_content no existe en el dataset -> NULL en BD.",
            "WTD se convierte de m a cm al cargar en flux_observations.",
            "timestamp diario anclado a 12:00 UTC en BD (UTC+1 solo en metadatos).",
        ],
        "outputs": {
            "observations_parquet": C.OUT_OBSERVATIONS.name,
            "ml_dataset_parquet": C.OUT_ML_DATASET.name,
        },
    }
    C.OUT_METADATA.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.add(
        "7-write",
        "Escritos parquet de observaciones, parquet ML y JSON de metadatos",
        observations_parquet=str(C.OUT_OBSERVATIONS.relative_to(C.PROJECT_ROOT)),
        ml_dataset_parquet=str(C.OUT_ML_DATASET.relative_to(C.PROJECT_ROOT)),
        metadata_json=str(C.OUT_METADATA.relative_to(C.PROJECT_ROOT)),
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    _logging.setup("etl")
    if not C.RAW_CSV.exists():
        raise SystemExit(f"No se encuentra el CSV crudo: {C.RAW_CSV}")

    log = CleaningLog()
    print("=" * 79)
    print("ETL DE-Zrk (Zarnekow) - flujos de carbono diarios")
    print("=" * 79)

    df = load_raw(log)                       # paso 1
    df = add_targets(df, log)                # pasos 2-3
    df = filter_period(df, log)              # paso 4
    df = interpolate_inputs(df, log)        # limpieza F
    observations = df.copy()                 # -> parquet 1 (1096 filas)

    df, feature_cols = engineer_features(df, log)   # paso 5
    ml = temporal_split(df, feature_cols, log)      # paso 6
    write_outputs(observations, ml, feature_cols, log)  # paso 7

    print("-" * 79)
    print("OK. Resumen:")
    print(f"  observaciones 2016-2018 : {len(observations):>5d} filas -> {C.OUT_OBSERVATIONS.name}")
    print(f"  matriz ML               : {len(ml):>5d} filas, {len(feature_cols)} features -> {C.OUT_ML_DATASET.name}")
    print(f"  train / test            : {(ml['split']=='train').sum()} / {(ml['split']=='test').sum()}")
    print(f"  metadatos               : {C.OUT_METADATA.name}")


if __name__ == "__main__":
    main()