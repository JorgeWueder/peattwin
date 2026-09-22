"""Metricas de regresion y clasificacion."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # NSE (Nash-Sutcliffe) y KGE (Kling-Gupta): metricas hidrologicas (esquema BD).
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    nse = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(y_pred.std() / y_true.std()) if y_true.std() > 0 else float("nan")
    # KGE no esta definido cuando la media observada ~ 0 (el termino beta se
    # dispara). Es el caso de NEE en DE-Zrk (media ~ 0.05, std ~ 1.08) -> nan.
    if abs(y_true.mean()) < 0.1 * y_true.std():
        kge = float("nan")
    else:
        beta = float(y_pred.mean() / y_true.mean())
        kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"rmse": rmse, "mae": mae, "r2": r2, "nse": nse, "kge": kge}


def classification_metrics(y_true, y_pred, y_score) -> dict:
    """Clase positiva = 1 (sumidero, minoritaria)."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "precision_fuente": float(p[0]),
        "recall_fuente": float(r[0]),
        "f1_fuente": float(f[0]),
        "precision_sumidero": float(p[1]),
        "recall_sumidero": float(r[1]),
        "f1_sumidero": float(f[1]),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def minmax_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)
