"""Carga del dataset procesado y construccion de la matriz de entrenamiento.

Sustituye TS_1..TS_5 por TS_promedio (media del perfil) y recalcula sus
lags/medias moviles sobre la serie diaria continua de observaciones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.training import config as C

NON_FEATURES = {
    "fluxnet_id", "timestamp", "split",
    C.TARGET_NEE, C.TARGET_FCH4, "clase_balance_carbono", "clase_balance_carbono_cod",
}


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    ml = pd.read_parquet(C.ML_PARQUET, engine="pyarrow")
    obs = pd.read_parquet(C.OBS_PARQUET, engine="pyarrow")
    ml["timestamp"] = pd.to_datetime(ml["timestamp"])
    obs["timestamp"] = pd.to_datetime(obs["timestamp"])

    # TS_promedio y derivadas sobre la serie diaria continua (obs: 1096 dias)
    obs = obs.sort_values("timestamp").reset_index(drop=True)
    obs[C.TS_MEAN] = obs[C.SOIL_TEMP_RAW].mean(axis=1)
    for lag in (1, 3, 7):
        obs[f"{C.TS_MEAN}_lag{lag}"] = obs[C.TS_MEAN].shift(lag)
    for w in (7, 30):
        obs[f"{C.TS_MEAN}_roll{w}_mean"] = obs[C.TS_MEAN].rolling(w, min_periods=w).mean()
    ts_cols = [
        C.TS_MEAN,
        f"{C.TS_MEAN}_lag1", f"{C.TS_MEAN}_lag3", f"{C.TS_MEAN}_lag7",
        f"{C.TS_MEAN}_roll7_mean", f"{C.TS_MEAN}_roll30_mean",
    ]

    df = ml.merge(obs[["timestamp", *ts_cols]], on="timestamp", how="left")
    df = df.drop(columns=C.DROP_FEATURES)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["clase_balance_carbono_cod"] = df["clase_balance_carbono_cod"].astype(int)

    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    na = df[feature_cols].isna().sum()
    if na.any():
        raise ValueError(f"NaN en features tras la construccion:\n{na[na > 0]}")
    return df, feature_cols


def train_test_frames(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    tr = df["split"].to_numpy() == "train"
    te = df["split"].to_numpy() == "test"
    return {
        "X_train": df.loc[tr, feature_cols], "X_test": df.loc[te, feature_cols],
        "nee_train": df.loc[tr, C.TARGET_NEE], "nee_test": df.loc[te, C.TARGET_NEE],
        "fch4_train": df.loc[tr, C.TARGET_FCH4], "fch4_test": df.loc[te, C.TARGET_FCH4],
        "cls_train": df.loc[tr, C.TARGET_CLS].astype(int),
        "cls_test": df.loc[te, C.TARGET_CLS].astype(int),
        "ts_train": df.loc[tr, "timestamp"], "ts_test": df.loc[te, "timestamp"],
    }


def make_sequences(df: pd.DataFrame, feature_cols: list[str], window: int):
    """Ventanas deslizantes de `window` dias -> objetivo del ultimo dia de la ventana."""
    X = df[feature_cols].to_numpy(np.float32)
    yreg = df[[C.TARGET_NEE, C.TARGET_FCH4]].to_numpy(np.float32)
    ycls = df[C.TARGET_CLS].to_numpy(np.float32)
    split = df["split"].to_numpy()
    ts = df["timestamp"].to_numpy()

    Xs, Yr, Yc, Sp, Ts = [], [], [], [], []
    for i in range(window - 1, len(df)):
        Xs.append(X[i - window + 1 : i + 1])
        Yr.append(yreg[i])
        Yc.append(ycls[i])
        Sp.append(split[i])
        Ts.append(ts[i])
    return np.stack(Xs), np.stack(Yr), np.asarray(Yc), np.asarray(Sp), np.asarray(Ts)
