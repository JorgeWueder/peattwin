"""Carga del modelo activo y construccion del vector de features del simulador.

NO reentrena: solo carga `modelo_final.pkl` (scikit-learn / stacking, via joblib)
o `modelo_final.h5` (Keras / CNN-LSTM, via tensorflow, import perezoso) segun la
extension del `file_path` del modelo activo en `ml_models`.

Construccion de features de un escenario  (documentado para el informe):

  * El escenario fija el NIVEL FREATICO objetivo. Por tanto `WTD` y todas sus
    derivadas (lags 1/3/7 d y medias moviles 7/30 d) se ponen al mismo valor
    objetivo -> hipotesis de "nivel freatico mantenido constante".
  * El resto de drivers (meteo, temperatura de suelo TS_promedio y sus derivadas)
    se toman de la CLIMATOLOGIA SUAVIZADA del dia del anio (media 2016-2018 del
    dataset de entrenamiento, ventana circular de 15 dias). Es decir: "un dia
    tipico de esa epoca del anio".
  * Las variables estacionales (doy, month y sus senos/cosenos) se calculan de la
    fecha pedida.

El resultado es determinista: la prediccion del modelo para un dia como el
indicado, con el nivel freatico llevado al valor del escenario y el resto de
condiciones en su climatologia estacional.
"""
from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.ml_model import MLModel

# Ventana del CNN-LSTM (debe coincidir con ml/training/config.py::WINDOW)
CNN_LSTM_WINDOW = 30

# Features cuyo valor se sustituye por el nivel freatico objetivo del escenario.
WTD_FEATURES = [
    "WTD", "WTD_lag1", "WTD_lag3", "WTD_lag7",
    "WTD_roll7_mean", "WTD_roll30_mean",
]

# RLock (reentrante): _build_climatology() llama a feature_frame() mientras ya
# tiene el lock tomado.
_lock = threading.RLock()
_model_cache: dict = {}
_clim_cache: dict = {}
_frame_cache: dict = {}


# --------------------------------------------------------------------- utilidades
def _repo_on_path() -> None:
    root = str(settings.project_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def feature_frame() -> tuple[pd.DataFrame, list[str]]:
    """Frame de entrenamiento (32 features + timestamp + objetivos), cacheado.

    Reutiliza `ml.training.data.build_dataset()` para garantizar EXACTAMENTE el
    mismo conjunto y orden de features con que se entreno el modelo (incl. la
    sustitucion TS_1..TS_5 -> TS_promedio y sus derivadas).
    """
    key = "frame"
    with _lock:
        cached = _frame_cache.get(key)
        if cached is None:
            _repo_on_path()
            from ml.training.data import build_dataset  # import perezoso (monorepo)

            df, feats = build_dataset()
            cached = (df, feats)
            _frame_cache.clear()
            _frame_cache[key] = cached
    df, feats = cached
    return df.copy(), list(feats)


# alias interno historico
_feature_frame = feature_frame


# ------------------------------------------------------------------- climatologia
def get_climatology() -> tuple[pd.DataFrame, list[str]]:
    key = "climatology"
    with _lock:
        cached = _clim_cache.get(key)
        if cached is None:
            cached = _build_climatology()
            _clim_cache.clear()
            _clim_cache[key] = cached
    return cached


def _build_climatology() -> tuple[pd.DataFrame, list[str]]:
    df, feats = _feature_frame()
    df = df.copy()
    df["doy"] = pd.to_datetime(df["timestamp"]).dt.dayofyear.clip(upper=365)
    clim = df.groupby("doy")[feats].mean()
    clim = clim.reindex(range(1, 366)).interpolate(limit_direction="both")
    # suavizado circular +/- 7 dias (triplicando la serie)
    ext = pd.concat([clim, clim, clim], ignore_index=True)
    sm = ext.rolling(15, center=True, min_periods=1).mean().iloc[365:730].reset_index(drop=True)
    sm.index = pd.Index(range(1, 366), name="doy")
    return sm, feats


def build_feature_row(
    clim: pd.DataFrame, feats: list[str], *, wtd_meters: float, on_date: date
) -> dict[str, float]:
    doy = min(on_date.timetuple().tm_yday, 365)
    month = on_date.month
    row = {k: float(v) for k, v in clim.loc[doy].to_dict().items()}

    for col in WTD_FEATURES:
        if col in row:
            row[col] = wtd_meters

    if "doy" in row:
        row["doy"] = float(doy)
    if "month" in row:
        row["month"] = float(month)
    if "doy_sin" in row:
        row["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    if "doy_cos" in row:
        row["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)
    if "month_sin" in row:
        row["month_sin"] = math.sin(2 * math.pi * month / 12)
    if "month_cos" in row:
        row["month_cos"] = math.cos(2 * math.pi * month / 12)

    return {k: float(row[k]) for k in feats}


# ----------------------------------------------------------------- modelo activo
@dataclass
class ActiveModel:
    kind: str                # "pkl" | "h5"
    name: str
    version: str
    path: str
    feature_names: list[str]
    regression_model_id: int | None
    classification_model_id: int | None
    _obj: object             # bundle dict (pkl) o (keras_model, x_scaler, y_scaler)

    def predict_row(self, row: dict[str, float]) -> tuple[float, float, float]:
        """Devuelve (co2, ch4, proba_sumidero)."""
        X = pd.DataFrame([row])[self.feature_names]
        if self.kind == "pkl":
            b = self._obj
            co2 = float(b["regressor_nee"].predict(X)[0])
            ch4 = float(b["regressor_fch4"].predict(X)[0])
            proba = float(b["classifier"].predict_proba(X)[0, 1])
            return co2, ch4, proba

        keras_model, x_scaler, y_scaler = self._obj
        xs = x_scaler.transform(X.to_numpy("float32"))
        # escenario constante: se repite la fila WINDOW veces -> (1, WINDOW, n)
        seq = np.repeat(xs, CNN_LSTM_WINDOW, axis=0)[None, :, :].astype("float32")
        out = keras_model.predict(seq, verbose=0)
        reg = y_scaler.inverse_transform(np.asarray(out["regresion"]))[0]
        proba = float(np.asarray(out["clasificacion"]).ravel()[0])
        return float(reg[0]), float(reg[1]), proba


def _resolve_path(db_path: str | None):
    from pathlib import Path

    if settings.MODEL_FILE:
        p = Path(settings.MODEL_FILE)
        if p.exists():
            return p
    if db_path:
        p = Path(db_path)
        if p.exists():
            return p
    for ext in (".pkl", ".h5"):
        cand = settings.models_dir / f"modelo_final{ext}"
        if cand.exists():
            return cand
    return None


def _load(path) -> tuple[str, object, list[str]]:
    if path.suffix == ".pkl":
        bundle = joblib.load(path)
        feats = list(bundle["regressor_nee"].feature_names_in_)
        return "pkl", bundle, feats

    # .h5 -> Keras/TensorFlow (import perezoso; solo si el modelo activo es CNN-LSTM)
    from tensorflow import keras  # noqa: PLC0415

    # compile=False: aqui solo se predice. Keras 3 no puede deserializar la
    # configuracion de perdida que guarda el .h5 heredado y falla con
    # "Could not deserialize 'keras.metrics.mse'" si se intenta compilar.
    keras_model = keras.models.load_model(path, compile=False)
    scalers = joblib.load(path.with_name("modelo_final_scalers.pkl"))
    _, feats = _feature_frame()
    return "h5", (keras_model, scalers["x_scaler"], scalers["y_scaler"]), feats


def load_active_model(db: Session) -> ActiveModel:
    reg = db.scalars(
        select(MLModel).where(MLModel.is_active.is_(True), MLModel.task == "REGRESSION")
    ).first()
    clf = db.scalars(
        select(MLModel).where(MLModel.is_active.is_(True), MLModel.task == "CLASSIFICATION")
    ).first()
    src = reg or clf
    path = _resolve_path(src.file_path if src else None)
    if path is None:
        raise FileNotFoundError(
            "No hay un modelo activo con archivo disponible. Ejecuta el Paso 5-8 "
            "(ml.training.run_training / tuning) o define MODEL_FILE."
        )

    key = (str(path), path.stat().st_mtime_ns)
    with _lock:
        cached = _model_cache.get(key)
        if cached is None:
            cached = _load(path)
            _model_cache.clear()
            _model_cache[key] = cached
    kind, obj, feats = cached

    return ActiveModel(
        kind=kind,
        name=(src.name.rsplit(" - ", 1)[0] if src else path.stem),
        version=(src.version if src else "n/a"),
        path=str(path),
        feature_names=feats,
        regression_model_id=(reg.id if reg else None),
        classification_model_id=(clf.id if clf else None),
        _obj=obj,
    )


def clear_caches() -> None:  # util para tests / recarga manual
    with _lock:
        _model_cache.clear()
        _clim_cache.clear()
        _frame_cache.clear()


def predict_frame(active: "ActiveModel", X: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    """Prediccion en lote sobre un DataFrame de features (columnas = active.feature_names)."""
    X = X[active.feature_names]
    if active.kind == "pkl":
        b = active._obj
        nee = [float(v) for v in b["regressor_nee"].predict(X)]
        fch4 = [float(v) for v in b["regressor_fch4"].predict(X)]
        proba = [float(v) for v in b["classifier"].predict_proba(X)[:, 1]]
        return nee, fch4, proba

    keras_model, x_scaler, y_scaler = active._obj
    xs = x_scaler.transform(X.to_numpy("float32"))
    seqs = np.stack([np.repeat(xs[i : i + 1], CNN_LSTM_WINDOW, axis=0) for i in range(len(xs))]).astype("float32")
    out = keras_model.predict(seqs, verbose=0)
    reg = y_scaler.inverse_transform(np.asarray(out["regresion"]))
    proba = np.asarray(out["clasificacion"]).ravel()
    return [float(v) for v in reg[:, 0]], [float(v) for v in reg[:, 1]], [float(v) for v in proba]
