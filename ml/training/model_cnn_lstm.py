"""Modelo hibrido CNN-LSTM multi-cabeza (Keras / TensorFlow).

Arquitectura:
    Input (WINDOW dias, n_features)
      -> Conv1D x2 (padding causal: no mira al futuro dentro de la ventana)
      -> MaxPooling1D                         [patrones locales de corto plazo]
      -> LSTM(64, seq) -> LSTM(32)            [dependencia temporal larga]
      -> Dropout -> Dense(32)                 [representacion compartida]
      -> cabeza "regresion"      : Dense(2) lineal   -> [NEE, FCH4]
      -> cabeza "clasificacion"  : Dense(1) sigmoide -> P(sumidero)

Perdida = MSE(regresion) + w * BCE(clasificacion), con w > 1 por el desbalance.
Los objetivos de regresion se estandarizan (el escalado se hace fuera, en
run_training) para que ambas cabezas operen en escalas comparables.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from ml.training import config as C  # noqa: E402


def set_seeds() -> None:
    keras.utils.set_random_seed(C.SEED)


def build_model(n_features: int, window: int) -> keras.Model:
    inp = keras.Input(shape=(window, n_features), name="ventana")
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    shared = layers.Dense(32, activation="relu")(x)

    out_reg = layers.Dense(2, name="regresion")(shared)               # [NEE, FCH4] estandarizados
    out_cls = layers.Dense(1, activation="sigmoid", name="clasificacion")(shared)

    # Salidas como dict -> loss/metrics/targets/sample_weight se mapean por nombre.
    model = keras.Model(inp, {"regresion": out_reg, "clasificacion": out_cls})
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"regresion": "mse", "clasificacion": "binary_crossentropy"},
        loss_weights={"regresion": 1.0, "clasificacion": C.CNN_LSTM_CLS_LOSS_WEIGHT},
        metrics={"clasificacion": [keras.metrics.AUC(name="auc")]},
    )
    return model
