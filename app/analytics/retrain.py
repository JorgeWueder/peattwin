"""Reentrenamiento real y sincrono de los 5 algoritmos desde la interfaz.

Orquesta las funciones que YA existen en `ml/training/run_training.py` —las
mismas que usa el script de linea de comandos—, de modo que el boton de la
seccion Motor IA y `python -m ml.training.run_training --save-all` producen
exactamente los mismos artefactos.

Se expone como GENERADOR: cada paso hace `yield Paso(porcentaje, mensaje)` y la
vista traduce eso a `st.progress` / `st.write`. Asi el dominio no importa
Streamlit y la logica se puede probar desde un script.

-----------------------------------------------------------------------------
TRADE-OFF ACEPTADO (entorno mono-usuario de desarrollo/demo):
no hay lock, ni cola, ni escritura atomica sobre `modelo_final.pkl`. Dos admins
reentrenando a la vez dejarian el artefacto inconsistente, y el predictor —que
cachea por (ruta, mtime)— podria recargar un modelo a medio escribir. Es una
decision consciente para un despliegue de un solo usuario, NO un descuido:
no usar este boton en un entorno con concurrencia.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd

from ml import _logging
from ml.training import config as C
from ml.training import data as D
from ml.training import run_training as RT

# Etiqueta visible -> nombre interno del pipeline. El 5º algoritmo del proyecto
# de referencia era LSTM-Autoencoder+RF; en PeatTwin el hibrido equivalente es
# el Stacking Ensemble (RF + XGBoost + SVR con meta-modelo Ridge).
ALGORITMOS = {
    "Random Forest": "Random Forest",
    "XGBoost": "Gradient Boosting (XGBoost)",
    "SVR / SVM": "SVR / SVM",
    "CNN-LSTM": "CNN-LSTM",
    "Stacking Ensemble": "Stacking Ensemble",
}

# Solo el CNN-LSTM necesita TensorFlow.
REQUIERE_TENSORFLOW = {"CNN-LSTM"}

_ORDEN_CSV = [
    "model", "algorithm_type", "framework", "train_time_s",
    "nee_rmse", "nee_mae", "nee_r2", "nee_nse", "nee_kge",
    "fch4_rmse", "fch4_mae", "fch4_r2", "fch4_nse", "fch4_kge",
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_sumidero", "recall_sumidero", "f1_sumidero",
    "precision_fuente", "recall_fuente", "f1_fuente",
    "auc", "tn", "fp", "fn", "tp",
]


@dataclass(frozen=True)
class Paso:
    """Un avance del reentrenamiento: porcentaje 0-100 y mensaje para la interfaz."""

    porcentaje: int
    mensaje: str
    resultado: pd.DataFrame | None = None
    ganador: str | None = None
    log: str | None = None


def tensorflow_disponible() -> bool:
    """Comprueba la presencia de TensorFlow SIN importarlo.

    `find_spec` mira el sistema de importacion y no ejecuta el paquete: evita los
    ~8 s de arranque de TF y, de paso, no introduce ningun nombre local que
    pudiera sombrear a uno global.
    """
    return importlib.util.find_spec("tensorflow") is not None


def ejecutar(
    algoritmos: list[str], *, guardar: bool = True, registrar_bd: bool = True
) -> Iterator[Paso]:
    """Reentrena los algoritmos indicados y va emitiendo el progreso.

    `algoritmos` son etiquetas de `ALGORITMOS`. Reproduce la secuencia de
    `run_training.main()`: cargar datos, preparar el holdout temporal, entrenar,
    evaluar, comparar y guardar.
    """
    fichero_log = _logging.setup("reentrenamiento_ui")

    yield Paso(0, "Cargando dataset DE-Zrk 2016-2018...")
    df, feats = D.build_dataset()

    yield Paso(10, f"Preparando holdout temporal 2016-17 / 2018 ({len(feats)} features)...")
    frames = D.train_test_frames(df, feats)
    spw = float((frames["cls_train"] == 0).sum() / (frames["cls_train"] == 1).sum())
    yield Paso(
        20,
        f"train={len(frames['X_train'])} · test={len(frames['X_test'])} · "
        f"scale_pos_weight(XGB)={spw:.3f}",
    )

    # --- entrenamiento y evaluacion (cada run_* evalua en el holdout) ---
    filas: list[dict] = []
    total = max(len(algoritmos), 1)
    for i, etiqueta in enumerate(algoritmos):
        nombre = ALGORITMOS[etiqueta]
        yield Paso(20 + int(i / total * 60), f"Entrenando {etiqueta}...")
        if nombre == "CNN-LSTM":
            filas.append(RT.run_cnn_lstm(df, feats))
        else:
            filas.append(RT.run_classical(nombre, frames, spw))
        yield Paso(20 + int((i + 1) / total * 60), f"{etiqueta} entrenado y evaluado.")

    yield Paso(85, "Comparando algoritmos y seleccionando el mejor...")
    res = RT.add_scores(pd.DataFrame(filas))
    ganador = str(res.loc[res["winner"], "model"].iloc[0])
    fila_ganadora = next(r for r in filas if r["model"] == ganador)

    if not guardar:
        yield Paso(
            100,
            f"Terminado sin guardar. Ganador de esta corrida: {ganador}. "
            "Los artefactos en disco no se han tocado.",
            resultado=res,
            ganador=ganador,
            log=str(fichero_log),
        )
        _logging.cerrar()
        return

    yield Paso(90, "Guardando tabla comparativa y figuras...")
    columnas = [c for c in _ORDEN_CSV if c in res.columns]
    res[columnas + ["reg_score_norm", "cls_score", "combined_score", "winner"]].to_csv(
        C.RESULTS_CSV, index=False
    )
    _generar_figuras(filas, frames, df, feats, res)

    yield Paso(95, f"Guardando los {len(filas)} modelos y el ganador ({ganador})...")
    ruta, formato = RT.save_winner(fila_ganadora)
    RT.save_all(filas)
    _escribir_metadata(res, fila_ganadora, ganador, feats, ruta, formato)
    RT.write_summary_md(res, feats, ganador)

    if registrar_bd:
        yield Paso(98, "Registrando en PostgreSQL (ml_models / ml_model_metrics)...")
        try:
            RT.register_in_db(res, ganador, ruta)
        except Exception as exc:  # la BD no debe tumbar el reentrenamiento
            yield Paso(98, f"AVISO: no se pudo registrar en la BD: {exc}")

    yield Paso(
        100,
        f"Reentrenamiento completo. Modelo desplegado: {ganador}.",
        resultado=res,
        ganador=ganador,
        log=str(fichero_log),
    )
    _logging.cerrar()


def _generar_figuras(
    filas: list[dict], frames: dict, df: pd.DataFrame, feats: list[str], res: pd.DataFrame
) -> None:
    """Regenera las figuras del Paso 5-6. Tolerante a fallo: no aborta la corrida."""
    try:
        _X, _Yr, Yc, Sp, _Ts = D.make_sequences(df, feats, C.WINDOW)
        cnn_true = Yc[Sp == "test"].astype(int)
        RT.figure_confusion(filas, frames, cnn_true)
        RT.figure_roc(filas, frames, cnn_true)
        RT.figure_heatmap(res)
        RT.figure_bars(res)
    except Exception as exc:  # pragma: no cover
        print(f"  AVISO: no se pudieron regenerar todas las figuras: {exc}")


def _escribir_metadata(
    res: pd.DataFrame, fila_ganadora: dict, ganador: str, feats: list[str],
    ruta: str, formato: str,
) -> None:
    """Mismo `modelo_final_metadata.json` que escribe el script de consola."""
    meta = {
        "winner": ganador,
        "criterio": "0.5*reg_score_norm + 0.5*(0.5*F1_macro + 0.5*AUC)",
        "combined_score": float(res.loc[res["winner"], "combined_score"].iloc[0]),
        "saved_as": ruta,
        "format": formato,
        "features": feats,
        "n_features": len(feats),
        "holdout": {"train_years": [2016, 2017], "test_year": 2018},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "origen": "reentrenamiento desde la seccion Motor IA",
        "metrics_winner": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in fila_ganadora.items()
            if k not in ("_fitted", "proba_sumidero")
        },
    }
    C.FINAL_META.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
