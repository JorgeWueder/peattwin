"""Prediccion interactiva de NEE y FCH4 con cualquiera de los modelos guardados.

Reutiliza la climatologia y el constructor de features de `app.ml.predictor`
—que ya implementa la hipotesis de "nivel freatico sostenido"— y la extiende a
otros drivers: si el usuario fija la temperatura del aire o del suelo, ese valor
se propaga tambien a sus lags y medias moviles, con el mismo criterio.

Los drivers que el usuario NO fija se toman de la climatologia del dia del anio
(media 2016-2018 suavizada), asi que el resultado es "un dia tipico de esa epoca
del anio, con tus valores impuestos".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml import predictor

# El dataset entreno con WTD en METROS; la interfaz trabaja en cm.
_CM_POR_M = 100.0

# Drivers que la interfaz permite sobrescribir, con sus derivadas.
# clave -> (etiqueta, unidad, columnas que reciben el mismo valor)
DRIVERS: dict[str, tuple[str, str, list[str]]] = {
    "TA_F": (
        "Temperatura del aire",
        "°C",
        ["TA_F", "TA_F_roll7_mean", "TA_F_roll30_mean"],
    ),
    "TS_promedio": (
        "Temperatura del suelo",
        "°C",
        ["TS_promedio", "TS_promedio_lag1", "TS_promedio_lag3", "TS_promedio_lag7",
         "TS_promedio_roll7_mean", "TS_promedio_roll30_mean"],
    ),
    "P_F": (
        "Precipitacion diaria",
        "mm",
        ["P_F"],          # las acumuladas roll7/roll30 se recalculan aparte
    ),
    "VPD_F": (
        "Deficit de presion de vapor",
        "hPa",
        ["VPD_F"],
    ),
}

# Acumulados de precipitacion: si se fija P_F, el acumulado coherente es N * P_F.
_P_ACUMULADOS = {"P_F_roll7_sum": 7, "P_F_roll30_sum": 30}


@dataclass(frozen=True)
class Resultado:
    nee: float
    fch4: float
    proba_sumidero: float
    clase: str
    clase_por_signo: str
    modelo: str
    formato: str
    supuestos: list[str]


def rangos_climatologia() -> pd.DataFrame:
    """Minimo, maximo y media de cada driver en el dataset, para acotar los sliders."""
    df, _feats = predictor.feature_frame()
    filas = []
    for clave, (etiqueta, unidad, _cols) in DRIVERS.items():
        if clave not in df.columns:
            continue
        serie = df[clave].astype(float)
        filas.append(
            {
                "clave": clave,
                "etiqueta": etiqueta,
                "unidad": unidad,
                "min": float(serie.min()),
                "max": float(serie.max()),
                "media": float(serie.mean()),
            }
        )
    return pd.DataFrame.from_records(filas)


def construir_fila(
    *, wtd_cm: float, on_date: date, overrides: dict[str, float] | None = None
) -> tuple[dict[str, float], list[str]]:
    """Vector de 32 features + la lista de supuestos que lo describen."""
    clim, feats = predictor.get_climatology()
    fila = predictor.build_feature_row(
        clim, feats, wtd_meters=wtd_cm / _CM_POR_M, on_date=on_date
    )

    supuestos = [
        f"Nivel freatico sostenido a {wtd_cm:.1f} cm ({wtd_cm / _CM_POR_M:.3f} m) "
        "en WTD y en sus lags y medias moviles.",
    ]

    aplicados = overrides or {}
    for clave, valor in aplicados.items():
        etiqueta, unidad, columnas = DRIVERS[clave]
        for col in columnas:
            if col in fila:
                fila[col] = float(valor)
        if clave == "P_F":
            for col, dias in _P_ACUMULADOS.items():
                if col in fila:
                    fila[col] = float(valor) * dias
        supuestos.append(
            f"{etiqueta} fijada a {valor:g} {unidad}, propagada a sus "
            f"{len(columnas)} columnas derivadas."
        )

    restantes = [k for k in DRIVERS if k not in aplicados]
    if restantes:
        etiquetas = ", ".join(DRIVERS[k][0].lower() for k in restantes)
        supuestos.append(
            f"Sin fijar ({etiquetas}) y el resto de drivers: climatologia suavizada "
            f"del dia del anio {on_date.timetuple().tm_yday} (media 2016-2018, "
            "ventana circular de 15 dias)."
        )
    supuestos.append(
        "Variables estacionales (doy, month y sus senos/cosenos) calculadas de la fecha."
    )
    supuestos.append(
        "El dataset de DE-Zrk no tiene indices de vegetacion (NDVI/LAI): la "
        "incertidumbre del NEE es alta (R2 holdout ~ 0.24)."
    )
    return fila, supuestos


def predecir(ruta_modelo: Path, fila: dict[str, float], supuestos: list[str]) -> Resultado:
    """Carga el bundle indicado y predice. No reentrena nada."""
    if ruta_modelo.suffix == ".h5":
        return _predecir_keras(ruta_modelo, fila, supuestos)

    bundle = joblib.load(ruta_modelo)
    reg_nee = bundle["regressor_nee"]
    X = pd.DataFrame([fila])[list(reg_nee.feature_names_in_)]
    nee = float(reg_nee.predict(X)[0])
    fch4 = float(bundle["regressor_fch4"].predict(X)[0])
    proba = float(bundle["classifier"].predict_proba(X)[0, 1])
    return _resultado(bundle.get("model_name", ruta_modelo.stem), "pkl", nee, fch4, proba, supuestos)


def _predecir_keras(ruta_modelo: Path, fila: dict[str, float], supuestos: list[str]) -> Resultado:
    """CNN-LSTM: el escenario constante se repite WINDOW dias para formar la ventana."""
    # Unico import diferido del proyecto, y deliberado: TensorFlow tarda ~8 s en
    # cargar y solo hace falta si el modelo elegido es un .h5. No puede provocar
    # el UnboundLocalError clasico porque `keras` no se usa a nivel de modulo en
    # este fichero: no hay ningun nombre global al que sombrear.
    from tensorflow import keras  # noqa: PLC0415

    # compile=False: solo se hace inferencia, nunca `fit`. Ademas Keras 3 no sabe
    # deserializar la configuracion de perdida guardada por el .h5 heredado
    # ("Could not deserialize 'keras.metrics.mse'"), y esa config no se necesita
    # para predecir.
    modelo = keras.models.load_model(ruta_modelo, compile=False)
    escaladores = joblib.load(ruta_modelo.with_name(ruta_modelo.stem + "_scalers.pkl"))
    _df, feats = predictor.feature_frame()
    X = pd.DataFrame([fila])[feats]
    xs = escaladores["x_scaler"].transform(X.to_numpy("float32"))
    seq = np.repeat(xs, predictor.CNN_LSTM_WINDOW, axis=0)[None, :, :].astype("float32")
    salida = modelo.predict(seq, verbose=0)
    reg = escaladores["y_scaler"].inverse_transform(np.asarray(salida["regresion"]))[0]
    proba = float(np.asarray(salida["clasificacion"]).ravel()[0])
    return _resultado("CNN-LSTM", "h5", float(reg[0]), float(reg[1]), proba, supuestos)


def _resultado(
    nombre: str, formato: str, nee: float, fch4: float, proba: float, supuestos: list[str]
) -> Resultado:
    return Resultado(
        nee=nee,
        fch4=fch4,
        proba_sumidero=proba,
        clase="sumidero" if proba >= 0.5 else "fuente",
        clase_por_signo="sumidero" if nee < 0 else "fuente",
        modelo=nombre,
        formato=formato,
        supuestos=supuestos + [f"Modelo cargado de disco sin reentrenar ({formato})."],
    )
