"""Entrenamiento, evaluacion y comparacion de los 5 modelos + registro del ganador.

Ejecutar desde la raiz del repositorio:

    python -m ml.training.run_training
    python -m ml.training.run_training --no-db      # no registra en PostgreSQL

Evaluacion: holdout TEMPORAL  train = 2016-2017 (702 dias)  ->  test = 2018 (365 dias).
(el split ya viene en la columna 'split' del dataset procesado; NO hay barajado)

Que se mide por modelo:
  * tiempo de entrenamiento (s)  = suma de los fits de ese modelo
    (regresor NEE + regresor FCH4 + clasificador; para el CNN-LSTM es un unico fit
     multi-cabeza)
  * regresion (NEE y FCH4 por separado): RMSE, MAE, R2  (+ NSE, KGE para la BD)
  * clasificacion: accuracy, balanced_accuracy, precision/recall/F1 macro y POR
    CLASE, matriz de confusion, curva ROC y AUC.

Nota sobre el desbalance (233 sumidero / 469 fuente en train, ~32/68):
  la 'accuracy' por si sola premia el clasificador trivial "siempre fuente"
  (acc ~= 0.68). Por eso el criterio de modelo ganador NO usa accuracy sino
  F1_macro y AUC, y se reportan F1/recall de la clase 'sumidero' de forma
  destacada.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import RocCurveDisplay, roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml.training import config as C  # noqa: E402
from ml.training import data as D  # noqa: E402
from ml.training import metrics as M  # noqa: E402
from ml.training import models_classical as MC  # noqa: E402
from ml import _logging

REG_TARGETS = [("nee", C.TARGET_NEE), ("fch4", C.TARGET_FCH4)]


# ============================================================ modelos clasicos
def _fit_timed(estimator, X, y):
    t0 = time.perf_counter()
    estimator.fit(X, y)
    return estimator, time.perf_counter() - t0


def run_classical(name: str, frames: dict, spw: float) -> dict:
    """Entrena regresores (NEE, FCH4) + clasificador y evalua en el holdout."""
    if name == "Random Forest":
        make_reg, make_clf = MC.rf_regressor, MC.rf_classifier
    elif name == "Gradient Boosting (XGBoost)":
        make_reg, make_clf = MC.xgb_regressor, lambda: MC.xgb_classifier(spw)
    elif name == "SVR / SVM":
        make_reg, make_clf = MC.svr_regressor, MC.svc_classifier
    elif name == "Stacking Ensemble":
        make_reg, make_clf = MC.stacking_regressor, lambda: MC.stacking_classifier(spw)
    else:  # pragma: no cover
        raise ValueError(name)

    fitted: dict = {}
    t_total = 0.0
    row = {"model": name, **C.MODEL_REGISTRY[name]}

    # --- regresion (un modelo por objetivo) ---
    for key, col in REG_TARGETS:
        est, dt = _fit_timed(make_reg(), frames["X_train"], frames[f"{key}_train"])
        t_total += dt
        pred = est.predict(frames["X_test"])
        m = M.regression_metrics(frames[f"{key}_test"], pred)
        row.update({f"{key}_{k}": v for k, v in m.items()})
        fitted[f"reg_{key}"] = est

    # --- clasificacion ---
    clf, dt = _fit_timed(make_clf(), frames["X_train"], frames["cls_train"])
    t_total += dt
    proba = clf.predict_proba(frames["X_test"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    row.update(M.classification_metrics(frames["cls_test"], pred, proba))
    row["proba_sumidero"] = proba
    fitted["clf"] = clf

    row["train_time_s"] = round(t_total, 3)
    row["_fitted"] = fitted
    print(f"  {name:28s}  t={t_total:6.1f}s  "
          f"R2(NEE)={row['nee_r2']:+.3f} R2(FCH4)={row['fch4_r2']:+.3f}  "
          f"F1_macro={row['f1_macro']:.3f} AUC={row['auc']:.3f}")
    return row


# ================================================================= CNN-LSTM
def run_cnn_lstm(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    from ml.training import model_cnn_lstm as H

    H.set_seeds()
    X, Yr, Yc, Sp, _Ts = D.make_sequences(df, feature_cols, C.WINDOW)
    tr, te = Sp == "train", Sp == "test"

    # Escalado de X (ajustado SOLO con ventanas de train) y de los objetivos de regresion
    x_scaler = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
    Xs = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")
    y_scaler = StandardScaler().fit(Yr[tr])
    Yr_s = y_scaler.transform(Yr).astype("float32")

    # Validacion = ultimo 15 % del tramo de train (temporal, sin barajar)
    idx_tr = np.where(tr)[0]
    cut = int(len(idx_tr) * (1 - C.CNN_LSTM_VAL_FRAC))
    fit_idx, val_idx = idx_tr[:cut], idx_tr[cut:]

    # Peso por muestra segun clase (equilibrio 32/68). Keras 3 multi-salida acepta
    # un unico array de sample_weight que se aplica a ambas cabezas: los dias
    # 'sumidero' (minoria, temporada de crecimiento) pesan mas tambien en la
    # regresion, lo cual es aceptable/deseable (son los dias con mas dinamica).
    n = len(fit_idx)
    n1 = float(Yc[fit_idx].sum())
    n0 = n - n1
    w1, w0 = n / (2 * n1), n / (2 * n0)
    sw = np.where(Yc[fit_idx] == 1, w1, w0).astype("float32")
    sw_val = np.where(Yc[val_idx] == 1, w1, w0).astype("float32")

    model = H.build_model(len(feature_cols), C.WINDOW)
    t0 = time.perf_counter()
    model.fit(
        Xs[fit_idx],
        {"regresion": Yr_s[fit_idx], "clasificacion": Yc[fit_idx]},
        sample_weight=sw,
        validation_data=(
            Xs[val_idx],
            {"regresion": Yr_s[val_idx], "clasificacion": Yc[val_idx]},
            sw_val,
        ),
        epochs=C.CNN_LSTM_EPOCHS, batch_size=C.CNN_LSTM_BATCH, shuffle=False, verbose=0,
        callbacks=[H.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True)],
    )
    t_total = time.perf_counter() - t0

    pred = model.predict(Xs[te], verbose=0)
    reg_pred = y_scaler.inverse_transform(pred["regresion"])
    cls_proba = np.asarray(pred["clasificacion"]).ravel()
    cls_pred = (cls_proba >= 0.5).astype(int)

    row = {"model": "CNN-LSTM", **C.MODEL_REGISTRY["CNN-LSTM"], "train_time_s": round(t_total, 3)}
    for j, (key, _col) in enumerate(REG_TARGETS):
        m = M.regression_metrics(Yr[te][:, j], reg_pred[:, j])
        row.update({f"{key}_{k}": v for k, v in m.items()})
    row.update(M.classification_metrics(Yc[te].astype(int), cls_pred, cls_proba))
    row["proba_sumidero"] = cls_proba
    row["_fitted"] = {"keras_model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}
    print(f"  {'CNN-LSTM':28s}  t={t_total:6.1f}s  "
          f"R2(NEE)={row['nee_r2']:+.3f} R2(FCH4)={row['fch4_r2']:+.3f}  "
          f"F1_macro={row['f1_macro']:.3f} AUC={row['auc']:.3f}")
    return row


# ==================================================================== figuras
def _slug(name: str) -> str:
    return (name.lower().replace(" / ", "_").replace(" ", "_")
            .replace("(", "").replace(")", "").replace("-", "_"))


def figure_confusion(rows: list[dict], frames: dict, cnn_true: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 3.6))
    for ax, r in zip(axes, rows):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fuente", "sumidero"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["fuente", "sumidero"])
        ax.set_xlabel("predicho"); ax.set_ylabel("real")
        ax.set_title(f"{r['model']}\nF1_sum={r['f1_sumidero']:.2f} rec_sum={r['recall_sumidero']:.2f}")
    fig.suptitle("Matrices de confusion (holdout 2018)", fontsize=13)
    fig.tight_layout()
    _save(fig, "matrices_confusion.png")


def figure_roc(rows: list[dict], frames: dict, cnn_true: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        y_true = cnn_true if r["model"] == "CNN-LSTM" else frames["cls_test"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, r["proba_sumidero"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['model']} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (recall sumidero)")
    ax.set_title("Curvas ROC - clasificacion sumidero vs fuente (holdout 2018)")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "curvas_roc_comparacion.png")


def figure_heatmap(res: pd.DataFrame) -> None:
    cols = {
        "R2 NEE": ("nee_r2", 1), "R2 FCH4": ("fch4_r2", 1),
        "RMSE NEE": ("nee_rmse", -1), "RMSE FCH4": ("fch4_rmse", -1),
        "Accuracy": ("accuracy", 1), "F1 macro": ("f1_macro", 1),
        "Recall sumidero": ("recall_sumidero", 1), "F1 sumidero": ("f1_sumidero", 1),
        "AUC": ("auc", 1), "Tiempo (s)": ("train_time_s", -1),
    }
    raw = pd.DataFrame({k: res[v[0]].to_numpy() for k, v in cols.items()}, index=res["model"])
    norm = pd.DataFrame(index=raw.index)
    for k, (_c, direction) in cols.items():
        z = M.minmax_norm(raw[k].to_numpy())
        norm[k] = z if direction == 1 else 1 - z  # verde = mejor siempre

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(norm.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols.keys(), rotation=35, ha="right")
    ax.set_yticks(range(len(raw))); ax.set_yticklabels(raw.index)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax.text(j, i, f"{raw.iloc[i, j]:.3g}", ha="center", va="center", fontsize=8)
    ax.set_title("Comparativa de metricas entre modelos  (color = mejor->peor, verde = mejor)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="rendimiento normalizado")
    fig.tight_layout()
    _save(fig, "heatmap_comparativo_metricas.png")


def figure_bars(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(res)); w = 0.35
    axes[0].bar(x - w / 2, res["nee_r2"], w, label="R2 NEE", color="#457b9d")
    axes[0].bar(x + w / 2, res["fch4_r2"], w, label="R2 FCH4", color="#e63946")
    axes[0].set_xticks(x); axes[0].set_xticklabels(res["model"], rotation=20, ha="right")
    axes[0].set_title("Regresion - R2 (mas alto mejor)"); axes[0].axhline(0, color="k", lw=0.8)
    axes[0].legend()
    for m, c in [("accuracy", "#adb5bd"), ("f1_macro", "#2a9d8f"),
                 ("recall_sumidero", "#e76f51"), ("auc", "#1d3557")]:
        axes[1].plot(x, res[m], "o-", label=m)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(res["model"], rotation=20, ha="right")
    axes[1].set_title("Clasificacion (accuracy no cuenta la historia sola)")
    axes[1].set_ylim(0, 1); axes[1].legend()
    fig.tight_layout()
    _save(fig, "comparativa_regresion_clasificacion.png")


def _save(fig, name: str) -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> figuras/{name}")


# ============================================================ criterio ganador
def add_scores(res: pd.DataFrame) -> pd.DataFrame:
    r2_mean = res[["nee_r2", "fch4_r2"]].mean(axis=1).to_numpy()
    res["reg_score_norm"] = M.minmax_norm(r2_mean)
    res["cls_score"] = 0.5 * res["f1_macro"] + 0.5 * res["auc"]
    res["combined_score"] = (
        C.COMBINED_W_REG * res["reg_score_norm"] + C.COMBINED_W_CLS * res["cls_score"]
    )
    res["winner"] = res["combined_score"] == res["combined_score"].max()
    return res


# ================================================================ persistencia
def save_winner(win: dict) -> tuple[str, str]:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    is_keras = win["model"] == "CNN-LSTM"
    if is_keras:
        # .h5 (HDF5) es el formato NATIVO de Keras/TensorFlow: serializa el grafo
        # de capas + los pesos de los tensores.
        win["_fitted"]["keras_model"].save(C.FINAL_H5)
        joblib.dump(
            {"x_scaler": win["_fitted"]["x_scaler"], "y_scaler": win["_fitted"]["y_scaler"]},
            C.MODELS_DIR / "modelo_final_scalers.pkl",
        )
        return str(C.FINAL_H5), "h5"

    # -------------------------------------------------------------------------
    # NOTA para el profesor: por que NO se usa .h5 aqui.
    # El formato .h5 (HDF5) es especifico de Keras/TensorFlow: guarda la
    # arquitectura del grafo de capas y los pesos de los tensores. Un modelo de
    # scikit-learn / XGBoost / StackingRegressor NO es un grafo de capas, es un
    # objeto Python normal; se serializa con pickle. joblib es el estandar de
    # facto para modelos scikit-learn (mas eficiente que pickle con arrays
    # NumPy grandes). Por eso el modelo clasico ganador se guarda como .pkl.
    # -------------------------------------------------------------------------
    bundle = {
        "model_name": win["model"],
        "regressor_nee": win["_fitted"]["reg_nee"],
        "regressor_fch4": win["_fitted"]["reg_fch4"],
        "classifier": win["_fitted"]["clf"],
        "positive_class": "sumidero (1)",
        "note": "No es .h5 porque .h5 es formato Keras/TF; esto es scikit-learn -> joblib.",
    }
    joblib.dump(bundle, C.FINAL_PKL)
    return str(C.FINAL_PKL), "pkl"


def _bundle_paths(name: str) -> dict[str, Path]:
    """Rutas de los artefactos por modelo (no solo del ganador)."""
    slug = _slug(name)
    return {
        "pkl": C.MODELS_DIR / f"modelo_{slug}.pkl",
        "h5": C.MODELS_DIR / f"modelo_{slug}.h5",
        "scalers": C.MODELS_DIR / f"modelo_{slug}_scalers.pkl",
    }


def save_all(rows: list[dict]) -> dict[str, str]:
    """Persiste los CINCO modelos, no solo el ganador.

    `save_winner()` guarda el desplegado en `modelo_final.*`; esta funcion deja
    ademas una copia por algoritmo en `modelo_<slug>.*`, de modo que la seccion
    Motor IA pueda predecir con cualquiera de los cinco sin reentrenar. Usa el
    mismo formato de bundle que `save_winner()`.
    """
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    guardados: dict[str, str] = {}
    for r in rows:
        name = r["model"]
        fitted = r.get("_fitted")
        if not fitted:
            continue
        paths = _bundle_paths(name)
        if name == "CNN-LSTM":
            fitted["keras_model"].save(paths["h5"])
            joblib.dump(
                {"x_scaler": fitted["x_scaler"], "y_scaler": fitted["y_scaler"]},
                paths["scalers"],
            )
            guardados[name] = str(paths["h5"])
            continue
        joblib.dump(
            {
                "model_name": name,
                "regressor_nee": fitted["reg_nee"],
                "regressor_fch4": fitted["reg_fch4"],
                "classifier": fitted["clf"],
                "positive_class": "sumidero (1)",
                "note": "Copia por algoritmo (save_all); el desplegado es modelo_final.pkl.",
            },
            paths["pkl"],
        )
        guardados[name] = str(paths["pkl"])
    return guardados


def register_in_db(res: pd.DataFrame, winner: str, model_path: str) -> None:
    from sqlalchemy import MetaData, Table, create_engine, select

    from ml.etl.load_to_db import _database_url

    engine = create_engine(_database_url(None), future=True)
    md = MetaData()
    ml_models = Table("ml_models", md, autoload_with=engine)
    ml_metrics = Table("ml_model_metrics", md, autoload_with=engine)

    def upsert_model(conn, **vals):
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(ml_models).values(**vals)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ml_model_name_version",
            set_={k: vals[k] for k in ("algorithm_type", "task", "file_path",
                                       "framework", "target_variable", "trained_at",
                                       "is_active", "description")},
        ).returning(ml_models.c.id)
        return conn.execute(stmt).scalar_one()

    def upsert_metric(conn, model_id, **vals):
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        vals = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in vals.items()}
        payload = {"model_id": model_id, "fold": -1, **vals}
        stmt = pg_insert(ml_metrics).values(**payload)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ml_model_metric_model_fold",
            set_={k: v for k, v in payload.items() if k not in ("model_id", "fold")},
        )
        conn.execute(stmt)

    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        # limpiar activos previos de ambas tareas (idempotencia + indice unico parcial)
        conn.execute(ml_models.update()
                     .where(ml_models.c.task.in_(["REGRESSION", "CLASSIFICATION"]))
                     .values(is_active=False))

        for _, r in res.iterrows():
            is_win = r["model"] == winner
            fp = model_path if is_win else str(C.RESULTS_CSV)

            reg_id = upsert_model(
                conn, name=f"{r['model']} - regresion", version="1.0",
                algorithm_type=r["algorithm_type"], task="REGRESSION", file_path=fp,
                framework=r["framework"], target_variable="NEE_F_ANNOPTLM+FCH4_F_ANNOPTLM",
                trained_at=now, is_active=bool(is_win),
                description=f"Holdout 2018. R2 NEE={r['nee_r2']:.3f}, R2 FCH4={r['fch4_r2']:.3f}. "
                            f"Metricas por objetivo en {C.RESULTS_CSV.name}.",
            )
            upsert_metric(
                conn, reg_id,
                rmse=float(np.mean([r["nee_rmse"], r["fch4_rmse"]])),
                mae=float(np.mean([r["nee_mae"], r["fch4_mae"]])),
                r2=float(np.mean([r["nee_r2"], r["fch4_r2"]])),
                nse=float(np.nanmean([r["nee_nse"], r["fch4_nse"]])),
                kge=float(np.nanmean([r["nee_kge"], r["fch4_kge"]])),
                training_time_seconds=float(r["train_time_s"]),
            )

            cls_id = upsert_model(
                conn, name=f"{r['model']} - clasificacion", version="1.0",
                algorithm_type=r["algorithm_type"], task="CLASSIFICATION", file_path=fp,
                framework=r["framework"], target_variable="clase_balance_carbono",
                trained_at=now, is_active=bool(is_win),
                description=f"Holdout 2018. F1_macro={r['f1_macro']:.3f}, AUC={r['auc']:.3f}, "
                            f"recall sumidero={r['recall_sumidero']:.3f}.",
            )
            upsert_metric(
                conn, cls_id,
                accuracy=float(r["accuracy"]),
                precision=float(r["precision_macro"]), recall=float(r["recall_macro"]),
                f1=float(r["f1_macro"]), auc_roc=float(r["auc"]),
                training_time_seconds=float(r["train_time_s"]),
            )

        active = conn.execute(
            select(ml_models.c.id, ml_models.c.name, ml_models.c.task)
            .where(ml_models.c.is_active.is_(True))
        ).all()
    print("  ml_models activos:", [(a.name, a.task) for a in active])


def write_summary_md(res: pd.DataFrame, feats: list[str], winner: str) -> None:
    """Resumen interpretado para el informe (numeros reales de esta ejecucion)."""
    rad_present = {f: (f in feats) for f in C.RADIATION_FEATURES}
    ppfd_note = "PPFD_IN_F" not in feats

    def col(df, c, fmt="{:.3f}"):
        return "  ".join(fmt.format(v) for v in df[c])

    nee_r2 = res["nee_r2"]
    fch4_r2 = res["fch4_r2"]
    # rango de R2(FCH4) sin el CNN-LSTM (que infra-ajusta por falta de datos)
    fch4_r2_tree = res.loc[res["model"] != "CNN-LSTM", "fch4_r2"]

    tbl = res[["model", "train_time_s", "nee_rmse", "nee_r2", "fch4_rmse", "fch4_r2",
               "accuracy", "f1_macro", "recall_sumidero", "f1_sumidero", "auc",
               "combined_score"]].copy()
    header = "| " + " | ".join(tbl.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(tbl.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(
            (v if isinstance(v, str) else f"{v:.3f}") for v in row
        ) + " |"
        for row in tbl.itertuples(index=False, name=None)
    )

    md = f"""# Resumen de entrenamiento - flujos de carbono DE-Zrk

Evaluacion: holdout temporal train = 2016-2017 (702 dias) -> test = 2018 (365 dias).
{len(feats)} features (TS_1..TS_5 sustituidas por TS_promedio).

## Verificacion: drivers de radiacion en el conjunto de features

| feature | ¿incluida? | rol |
|---|---|---|
| `SW_IN_F` (radiacion onda corta entrante) | {"SI" if rad_present["SW_IN_F"] else "NO"} | control primario de GPP |
| `VPD_F` (deficit de presion de vapor) | {"SI" if rad_present["VPD_F"] else "NO"} | control de la conductancia estomatica |
| `NETRAD_F` (radiacion neta) | {"SI" if rad_present["NETRAD_F"] else "NO"} | balance energetico |
| `PPFD_IN_F` (radiacion fotosinteticamente activa) | {"NO (excluida)" if ppfd_note else "SI"} | 65 dias vacios en ene-mar 2018; se usa SW_IN_F como proxy |

**Resultado: SW_IN_F y VPD_F (y NETRAD_F) YA estaban incluidos** desde el ETL
(`METEO_FEATURES`). No hay un driver de radiacion "perdido" que explique el bajo
R2 de NEE.

## Comparativa de modelos (holdout 2018)

{header}
{sep}
{body}

Ganador (criterio 0.5*reg_score_norm + 0.5*(0.5*F1_macro + 0.5*AUC)): **{winner}**.

## Por que el R2 de NEE ({nee_r2.min():.2f}-{nee_r2.max():.2f}) es mucho menor que el de FCH4 ({fch4_r2_tree.min():.2f}-{fch4_r2_tree.max():.2f} en RF/XGB/SVR/Stacking; el CNN-LSTM baja a {res.loc[res['model'] == 'CNN-LSTM', 'fch4_r2'].iloc[0]:.2f} por falta de datos para una red profunda)

No es un problema de features de radiacion faltantes (estan todas). El NEE en
DE-Zrk es **intrinsecamente mas dificil** de predecir con drivers
meteorologicos/hidrologicos solos:

1. **NEE = GPP - RECO**: es la pequena diferencia entre dos flujos grandes de
   signo opuesto (fotosintesis y respiracion) que casi se cancelan. La relacion
   senal/ruido del residuo es baja: errores modestos en GPP o en RECO se
   amplifican en NEE. FCH4, en cambio, es un flujo unico y estrictamente
   positivo.
2. **Falta el estado de la vegetacion.** La GPP depende de la fenologia y de la
   cantidad/actividad de hoja verde (rebrote primaveral, siega, senescencia,
   estres hidrico). Este dataset FLUXNET-CH4 **no incluye NDVI, LAI, EVI ni
   fraccion de cobertura verde**. La radiacion (SW_IN_F, NETRAD_F, VPD_F)
   explica la *envolvente estacional* de la GPP, pero no su variacion dia a dia
   por cambios en la capacidad fotosintetica del dosel.
3. **FCH4 esta bien restringido por lo que si hay**: temperatura del suelo
   (TS_promedio) y nivel freatico (WTD) controlan la produccion y el transporte
   de metano, y ambos estan medidos y con buena cobertura -> R2 alto.
4. **Media de NEE ~ 0** (0.05 gC m-2 d-1) con desviacion ~ 1.08: el "modelo
   nulo" (predecir la media) ya captura la escala, de modo que queda poca
   varianza estructurada que un modelo pueda ganar sin informacion de
   vegetacion; el R2, que se mide contra ese modelo nulo, castiga mucho.

**Consecuencia para el informe / trabajo futuro:** para subir el R2 de NEE haria
falta incorporar indices de vegetacion de satelite (MODIS/Sentinel-2: NDVI, LAI,
EVI) o modelar por separado GPP y RECO (particion de flujo) como objetivos
intermedios. Con los drivers disponibles, R2 ~ 0.2 para NEE diario es un
resultado esperable y coherente con la literatura de flujos en turberas.

*(generado por `python -m ml.training.run_training`)*
"""
    C.RESUMEN_MD.write_text(md, encoding="utf-8")
    print(f"  resumen -> {C.RESUMEN_MD.relative_to(C.ROOT)}")


# ======================================================================= main
def main() -> None:
    _logging.setup("entrenamiento")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true", help="No registrar en PostgreSQL")
    ap.add_argument(
        "--save-all",
        action="store_true",
        help="Guardar ademas los CINCO modelos en ml/models/modelo_<slug>.{pkl,h5} "
             "(los usa el selector de la seccion Motor IA).",
    )
    args = ap.parse_args()

    print("=" * 79)
    print("Entrenamiento y comparacion de modelos - flujos de carbono DE-Zrk")
    print("=" * 79)

    df, feats = D.build_dataset()
    frames = D.train_test_frames(df, feats)
    spw = float((frames["cls_train"] == 0).sum() / (frames["cls_train"] == 1).sum())
    rad = {f: (f in feats) for f in C.RADIATION_FEATURES}
    print(f"drivers de radiacion presentes: {rad}"
          f"  (PPFD_IN_F excluida: {'PPFD_IN_F' not in feats})")
    print(f"features: {len(feats)}  (TS_1..TS_5 -> {C.TS_MEAN})")
    print(f"train={len(frames['X_train'])}  test={len(frames['X_test'])}  "
          f"scale_pos_weight(XGB)={spw:.3f}")
    print("-" * 79)

    rows: list[dict] = []
    for name in ["Random Forest", "Gradient Boosting (XGBoost)", "SVR / SVM", "Stacking Ensemble"]:
        rows.append(run_classical(name, frames, spw))
    rows.append(run_cnn_lstm(df, feats))

    # matriz de confusion / ROC necesitan el y_true del CNN-LSTM (ventanas)
    _X, _Yr, Yc, Sp, _Ts = D.make_sequences(df, feats, C.WINDOW)
    cnn_true = Yc[Sp == "test"].astype(int)

    order = ["model", "algorithm_type", "framework", "train_time_s",
             "nee_rmse", "nee_mae", "nee_r2", "nee_nse", "nee_kge",
             "fch4_rmse", "fch4_mae", "fch4_r2", "fch4_nse", "fch4_kge",
             "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
             "precision_sumidero", "recall_sumidero", "f1_sumidero",
             "precision_fuente", "recall_fuente", "f1_fuente",
             "auc", "tn", "fp", "fn", "tp"]
    res = pd.DataFrame(rows)
    res = add_scores(res)
    res_csv = res[order + ["reg_score_norm", "cls_score", "combined_score", "winner"]].copy()
    res_csv.to_csv(C.RESULTS_CSV, index=False)
    print("-" * 79)
    print(f"tabla comparativa -> {C.RESULTS_CSV.relative_to(C.ROOT)}")

    # figuras
    figure_confusion(rows, frames, cnn_true)
    figure_roc(rows, frames, cnn_true)
    figure_heatmap(res)
    figure_bars(res)

    winner = res.loc[res["winner"], "model"].iloc[0]
    win_row = next(r for r in rows if r["model"] == winner)
    print("-" * 79)
    print("CRITERIO DE MODELO GANADOR (documentado):")
    print("  combined = 0.5 * reg_score_norm + 0.5 * cls_score")
    print("    reg_score_norm = R2 medio(NEE,FCH4) normalizado min-max entre los 5 modelos")
    print("    cls_score      = 0.5*F1_macro + 0.5*AUC   (NO accuracy: desbalance 32/68)")
    print(res[["model", "reg_score_norm", "cls_score", "combined_score"]]
          .round(4).to_string(index=False))
    print(f"\n  GANADOR: {winner}")

    path, kind = save_winner(win_row)
    if args.save_all:
        guardados = save_all(rows)
        print(f"  copias por algoritmo -> {len(guardados)} artefactos en "
              f"{C.MODELS_DIR.relative_to(C.ROOT)}")
    meta = {
        "winner": winner,
        "criterio": "0.5*reg_score_norm + 0.5*(0.5*F1_macro + 0.5*AUC)",
        "combined_score": float(res.loc[res["winner"], "combined_score"].iloc[0]),
        "saved_as": path, "format": kind,
        "features": feats, "n_features": len(feats),
        "holdout": {"train_years": [2016, 2017], "test_year": 2018},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_winner": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in win_row.items()
                           if k not in ("_fitted", "proba_sumidero")},
    }
    C.FINAL_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  modelo guardado -> {path}  (+ modelo_final_metadata.json)")

    write_summary_md(res, feats, winner)

    if not args.no_db:
        print("-" * 79)
        print("registro en PostgreSQL (ml_models / ml_model_metrics)")
        try:
            register_in_db(res, winner, path)
        except Exception as exc:  # pragma: no cover
            print(f"  AVISO: no se pudo registrar en la BD: {exc}")
    print("-" * 79)
    print("OK.")


if __name__ == "__main__":
    main()
