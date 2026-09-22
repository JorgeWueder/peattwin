"""Validacion cruzada TEMPORAL de los 5 modelos (RF, XGBoost, SVR/SVM, CNN-LSTM, Stacking).

Usa `sklearn.model_selection.TimeSeriesSplit` (NUNCA KFold aleatorio: barajar el
orden de una serie temporal es fuga de datos - el modelo "veria" el futuro).

Cada fold: train = [0 .. k], test = (k .. k+m], ambos CONTIGUOS y en orden. Los
folds sucesivos amplian la ventana de entrenamiento y desplazan la de test.

Para cada modelo y cada fold calcula:
  * regresion (NEE y FCH4, y su media): RMSE, MAE, R2, NSE, KGE
  * clasificacion (sumidero vs fuente): accuracy, precision/recall/F1 macro, AUC
  * tiempo de entrenamiento (s)

Salidas:
  ml/training/cv_por_fold.csv   -> una fila por (modelo, fold)
  ml/training/cv_resumen.csv    -> media y desv. estandar por modelo
  ml/training/figuras/cv_boxplots_estabilidad.png
  ml_model_metrics              -> filas fold = 0 .. n_splits-1 (upsert idempotente)

Ejecutar desde la raiz del repositorio:
    python -m ml.training.cross_validation
    python -m ml.training.cross_validation --folds 8 --gap 30
    python -m ml.training.cross_validation --no-db

NOTA: esta CV evalua la ESTABILIDAD de cada modelo sobre toda la serie 2016-2018
y es complementaria al holdout unico de run_training.py (train 2016-2017 / test
2018), que se guarda como fold = -1.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml.training import config as C  # noqa: E402
from ml.training import data as D  # noqa: E402
from ml.training import metrics as M  # noqa: E402
from ml.training import models_classical as MC  # noqa: E402
from ml import _logging

REG = [("nee", C.TARGET_NEE), ("fch4", C.TARGET_FCH4)]
CLASSICAL = ["Random Forest", "Gradient Boosting (XGBoost)", "SVR / SVM", "Stacking Ensemble"]
ALL_MODELS = CLASSICAL + ["CNN-LSTM"]

# metricas que se resumen (media +/- std) y su nombre "bonito"
SUMMARY_METRICS = [
    "rmse_nee", "rmse_fch4", "mae_nee", "mae_fch4",
    "r2_nee", "r2_fch4", "nse_nee", "nse_fch4", "kge_nee", "kge_fch4",
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc",
    "train_time_s",
]


# ------------------------------------------------------------- modelos clasicos
def _builders(name: str, spw: float):
    if name == "Random Forest":
        return MC.rf_regressor, MC.rf_classifier
    if name == "Gradient Boosting (XGBoost)":
        return MC.xgb_regressor, lambda: MC.xgb_classifier(spw)
    if name == "SVR / SVM":
        return MC.svr_regressor, MC.svc_classifier
    if name == "Stacking Ensemble":
        return MC.stacking_regressor, lambda: MC.stacking_classifier(spw)
    raise ValueError(name)


def eval_classical_fold(name: str, df: pd.DataFrame, feats: list[str],
                        tr: np.ndarray, te: np.ndarray) -> dict:
    Xtr, Xte = df[feats].iloc[tr], df[feats].iloc[te]
    ctr = df[C.TARGET_CLS].iloc[tr].astype(int)
    cte = df[C.TARGET_CLS].iloc[te].astype(int)
    spw = float((ctr == 0).sum() / max((ctr == 1).sum(), 1))
    make_reg, make_clf = _builders(name, spw)

    out: dict = {"model": name}
    t_total = 0.0
    for key, col in REG:
        est = make_reg()
        t0 = time.perf_counter()
        est.fit(Xtr, df[col].iloc[tr])
        t_total += time.perf_counter() - t0
        m = M.regression_metrics(df[col].iloc[te], est.predict(Xte))
        out.update({f"{k}_{key}": v for k, v in m.items()})

    clf = make_clf()
    t0 = time.perf_counter()
    clf.fit(Xtr, ctr)
    t_total += time.perf_counter() - t0
    proba = clf.predict_proba(Xte)[:, 1]
    out.update(M.classification_metrics(cte, (proba >= 0.5).astype(int), proba))
    out["train_time_s"] = t_total
    return out


# -------------------------------------------------------------------- CNN-LSTM
def _windows(df: pd.DataFrame, feats: list[str], window: int):
    X = df[feats].to_numpy(np.float32)
    yreg = df[[C.TARGET_NEE, C.TARGET_FCH4]].to_numpy(np.float32)
    ycls = df[C.TARGET_CLS].to_numpy(np.float32)
    end = np.arange(window - 1, len(df))
    Xs = np.stack([X[i - window + 1 : i + 1] for i in end])
    return Xs, yreg[end], ycls[end], end


def eval_cnn_lstm_fold(win, tr: np.ndarray, te: np.ndarray, feats: list[str]) -> dict | None:
    from ml.training import model_cnn_lstm as H

    Xs, Yr, Yc, end = win
    tr_m = np.isin(end, tr)
    te_m = np.isin(end, te)
    if tr_m.sum() < 60 or te_m.sum() < 15:
        return None  # fold demasiado corto para una red profunda

    H.set_seeds()
    nfeat = Xs.shape[-1]
    x_scaler = StandardScaler().fit(Xs[tr_m].reshape(-1, nfeat))
    Xg = x_scaler.transform(Xs.reshape(-1, nfeat)).reshape(Xs.shape).astype("float32")
    y_scaler = StandardScaler().fit(Yr[tr_m])
    Yg = y_scaler.transform(Yr).astype("float32")

    idx_tr = np.where(tr_m)[0]
    cut = max(int(len(idx_tr) * (1 - C.CNN_LSTM_VAL_FRAC)), len(idx_tr) - 60)
    fit_i, val_i = idx_tr[:cut], idx_tr[cut:]

    n = len(fit_i)
    n1 = max(float(Yc[fit_i].sum()), 1.0)
    n0 = max(n - n1, 1.0)
    w1, w0 = n / (2 * n1), n / (2 * n0)
    sw = np.where(Yc[fit_i] == 1, w1, w0).astype("float32")
    sw_val = np.where(Yc[val_i] == 1, w1, w0).astype("float32")

    model = H.build_model(nfeat, C.WINDOW)
    t0 = time.perf_counter()
    model.fit(
        Xg[fit_i], {"regresion": Yg[fit_i], "clasificacion": Yc[fit_i]}, sample_weight=sw,
        validation_data=(Xg[val_i], {"regresion": Yg[val_i], "clasificacion": Yc[val_i]}, sw_val),
        epochs=C.CNN_LSTM_EPOCHS, batch_size=C.CNN_LSTM_BATCH, shuffle=False, verbose=0,
        callbacks=[H.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                   restore_best_weights=True)],
    )
    t_total = time.perf_counter() - t0

    pred = model.predict(Xg[te_m], verbose=0)
    reg_pred = y_scaler.inverse_transform(pred["regresion"])
    proba = np.asarray(pred["clasificacion"]).ravel()

    out: dict = {"model": "CNN-LSTM"}
    for j, (key, _c) in enumerate(REG):
        m = M.regression_metrics(Yr[te_m][:, j], reg_pred[:, j])
        out.update({f"{k}_{key}": v for k, v in m.items()})
    out.update(M.classification_metrics(Yc[te_m].astype(int), (proba >= 0.5).astype(int), proba))
    out["train_time_s"] = t_total
    return out


# ----------------------------------------------------------------- resumen/figura
def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    g = per_fold.groupby("model")
    rows = {}
    for name in ALL_MODELS:
        if name not in g.groups:
            continue
        sub = g.get_group(name)
        rec = {"n_folds": len(sub)}
        for met in SUMMARY_METRICS:
            rec[f"{met}_mean"] = float(np.nanmean(sub[met]))
            rec[f"{met}_std"] = float(np.nanstd(sub[met], ddof=1)) if len(sub) > 1 else 0.0
        rows[name] = rec
    out = pd.DataFrame(rows).T
    out.index.name = "model"
    return out


def _fmt(mean: float, std: float, dec: int = 3) -> str:
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def print_summary(summary: pd.DataFrame) -> None:
    show = ["rmse_nee", "rmse_fch4", "r2_nee", "r2_fch4", "nse_fch4", "kge_fch4",
            "accuracy", "f1_macro", "auc", "train_time_s"]
    tbl = pd.DataFrame(
        {m: [_fmt(summary.loc[i, f"{m}_mean"], summary.loc[i, f"{m}_std"],
                  1 if m == "train_time_s" else 3) for i in summary.index]
         for m in show},
        index=summary.index,
    )
    print(tbl.to_string())


def boxplots(per_fold: pd.DataFrame) -> None:
    order = [m for m in ALL_MODELS if m in per_fold["model"].unique()]
    panels = [
        ("rmse_nee", "RMSE  NEE (CO2)  [gC m-2 d-1]  - mas bajo mejor"),
        ("rmse_fch4", "RMSE  FCH4 (CH4)  [nmol m-2 s-1]  - mas bajo mejor"),
        ("f1_macro", "F1 macro (clasificacion)  - mas alto mejor"),
        ("r2_fch4", "R2  FCH4  - mas alto mejor"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (col, title) in zip(axes.ravel(), panels):
        data = [per_fold.loc[per_fold["model"] == m, col].dropna().to_numpy() for m in order]
        bp = ax.boxplot(data, tick_labels=order, showmeans=True)
        for i, d in enumerate(data, start=1):
            ax.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)),
                       d, s=18, alpha=0.6, color="#1d3557", zorder=3)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(
        f"Estabilidad entre folds (TimeSeriesSplit, {per_fold['fold'].nunique()} folds) - "
        "caja mas corta = modelo mas estable", fontsize=13,
    )
    fig.tight_layout()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / "cv_boxplots_estabilidad.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> figuras/cv_boxplots_estabilidad.png")


# --------------------------------------------------------------------- registro BD
def register_in_db(per_fold: pd.DataFrame) -> None:
    from sqlalchemy import MetaData, Table, create_engine, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from ml.etl.load_to_db import _database_url

    engine = create_engine(_database_url(None), future=True)
    md = MetaData()
    ml_models = Table("ml_models", md, autoload_with=engine)
    ml_metrics = Table("ml_model_metrics", md, autoload_with=engine)

    def model_id(conn, name: str, task: str, target: str) -> int:
        mid = conn.execute(
            select(ml_models.c.id).where(ml_models.c.name == name,
                                         ml_models.c.version == "1.0")
        ).scalar_one_or_none()
        if mid is not None:
            return mid
        algo = C.MODEL_REGISTRY[name.rsplit(" - ", 1)[0]]
        return conn.execute(
            pg_insert(ml_models).values(
                name=name, version="1.0", algorithm_type=algo["algorithm_type"],
                task=task, file_path=str(C.CV_PER_FOLD_CSV), framework=algo["framework"],
                target_variable=target, is_active=False,
                description="Creado por cross_validation.py",
            ).returning(ml_models.c.id)
        ).scalar_one()

    def upsert_metric(conn, mid: int, fold: int, **vals):
        vals = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in vals.items()}
        stmt = pg_insert(ml_metrics).values(model_id=mid, fold=fold, **vals)
        conn.execute(stmt.on_conflict_do_update(
            constraint="uq_ml_model_metric_model_fold",
            set_={k: v for k, v in vals.items()},
        ))

    with engine.begin() as conn:
        for _, r in per_fold.iterrows():
            base = r["model"]
            reg_id = model_id(conn, f"{base} - regresion", "REGRESSION",
                              "NEE_F_ANNOPTLM+FCH4_F_ANNOPTLM")
            cls_id = model_id(conn, f"{base} - clasificacion", "CLASSIFICATION",
                              "clase_balance_carbono")
            fold = int(r["fold"])
            upsert_metric(
                conn, reg_id, fold,
                rmse=float(np.nanmean([r["rmse_nee"], r["rmse_fch4"]])),
                mae=float(np.nanmean([r["mae_nee"], r["mae_fch4"]])),
                r2=float(np.nanmean([r["r2_nee"], r["r2_fch4"]])),
                nse=float(np.nanmean([r["nse_nee"], r["nse_fch4"]])),
                kge=float(np.nanmean([r["kge_nee"], r["kge_fch4"]])),
                training_time_seconds=float(r["train_time_s"]),
            )
            upsert_metric(
                conn, cls_id, fold,
                accuracy=float(r["accuracy"]), precision=float(r["precision_macro"]),
                recall=float(r["recall_macro"]), f1=float(r["f1_macro"]),
                auc_roc=float(r["auc"]), training_time_seconds=float(r["train_time_s"]),
            )
        total = conn.execute(
            select(ml_metrics.c.id).where(ml_metrics.c.fold >= 0)
        ).all()
    print(f"  ml_model_metrics: {len(total)} filas con fold >= 0 (CV)")


# --------------------------------------------------------------------------- main
def main() -> None:
    _logging.setup("validacion_cruzada")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", type=int, default=C.CV_N_SPLITS, help="nº de folds (def. 5)")
    ap.add_argument("--gap", type=int, default=C.CV_GAP,
                    help="dias descartados entre train y test de cada fold (def. 0)")
    ap.add_argument("--no-db", action="store_true", help="no escribir en ml_model_metrics")
    ap.add_argument("--no-cnn", action="store_true", help="omitir CNN-LSTM (mas rapido)")
    args = ap.parse_args()

    print("=" * 79)
    print(f"Validacion cruzada temporal - TimeSeriesSplit(n_splits={args.folds}, gap={args.gap})")
    print("=" * 79)

    df, feats = D.build_dataset()
    n = len(df)
    tscv = TimeSeriesSplit(n_splits=args.folds, gap=args.gap)
    win = _windows(df, feats, C.WINDOW)

    per_fold_rows: list[dict] = []
    for fold, (tr, te) in enumerate(tscv.split(np.arange(n))):
        print(f"\n-- fold {fold}:  train[{tr[0]}..{tr[-1]}] ({len(tr)})   "
              f"test[{te[0]}..{te[-1]}] ({len(te)})   "
              f"{df['timestamp'].iloc[te[0]].date()} .. {df['timestamp'].iloc[te[-1]].date()}")
        for name in CLASSICAL:
            row = eval_classical_fold(name, df, feats, tr, te)
            row["fold"] = fold
            per_fold_rows.append(row)
            print(f"   {name:28s} RMSE(FCH4)={row['rmse_fch4']:7.2f}  "
                  f"R2(NEE)={row['r2_nee']:+.3f} R2(FCH4)={row['r2_fch4']:+.3f}  "
                  f"F1={row['f1_macro']:.3f} AUC={row['auc']:.3f}")
        if not args.no_cnn:
            row = eval_cnn_lstm_fold(win, tr, te, feats)
            if row is None:
                print("   CNN-LSTM                      (fold demasiado corto, se omite)")
            else:
                row["fold"] = fold
                per_fold_rows.append(row)
                print(f"   {'CNN-LSTM':28s} RMSE(FCH4)={row['rmse_fch4']:7.2f}  "
                      f"R2(NEE)={row['r2_nee']:+.3f} R2(FCH4)={row['r2_fch4']:+.3f}  "
                      f"F1={row['f1_macro']:.3f} AUC={row['auc']:.3f}")

    per_fold = pd.DataFrame(per_fold_rows)
    keep = (["model", "fold"]
            + [f"{k}_{t}" for t in ("nee", "fch4")
               for k in ("rmse", "mae", "r2", "nse", "kge")]
            + ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
               "f1_macro", "f1_sumidero", "recall_sumidero", "auc",
               "tn", "fp", "fn", "tp", "train_time_s"])
    per_fold = per_fold[[c for c in keep if c in per_fold.columns]]
    per_fold.to_csv(C.CV_PER_FOLD_CSV, index=False)

    summary = summarise(per_fold)
    summary.to_csv(C.CV_RESUMEN_CSV)

    print("\n" + "=" * 79)
    print(f"RESUMEN  (media ± desv. estandar sobre {args.folds} folds)")
    print("=" * 79)
    print_summary(summary)
    print("-" * 79)
    print(f"por fold  -> {C.CV_PER_FOLD_CSV.relative_to(C.ROOT)}")
    print(f"resumen   -> {C.CV_RESUMEN_CSV.relative_to(C.ROOT)}")

    boxplots(per_fold)

    if not args.no_db:
        print("-" * 79)
        print("registro en PostgreSQL (ml_model_metrics, fold 0 .. n-1)")
        try:
            register_in_db(per_fold)
        except Exception as exc:  # pragma: no cover
            print(f"  AVISO: no se pudo registrar en la BD: {exc}")
    print("-" * 79)
    print("OK.")


if __name__ == "__main__":
    main()
