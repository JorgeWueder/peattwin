"""Metricas del modelo desplegado con intervalos de confianza (bootstrap por bloques).

Carga `ml/models/modelo_final.pkl` (Stacking Ensemble ajustado), predice el holdout
2018 sin reentrenar, comprueba que las metricas coinciden con las guardadas en
`modelo_final_metadata.json` y calcula intervalos de confianza del 95 % por
bootstrap de bloques moviles (los dias consecutivos estan autocorrelacionados, asi
que remuestrear dias sueltos subestimaria la incertidumbre).

    python -m ml.training.metricas_modelo_desplegado --block 14 --reps 2000
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score

from ml.training import config as C
from ml.training import data as D

MODEL = C.ROOT / "ml" / "models" / "modelo_final.pkl"
META = C.ROOT / "ml" / "models" / "modelo_final_metadata.json"


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices de un remuestreo por bloques moviles (circular) de longitud n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def metrics(y_nee, p_nee, y_ch4, p_ch4, y_cls, proba) -> dict:
    out = {
        "NEE RMSE": float(np.sqrt(mean_squared_error(y_nee, p_nee))),
        "NEE R2": float(r2_score(y_nee, p_nee)),
        "FCH4 RMSE": float(np.sqrt(mean_squared_error(y_ch4, p_ch4))),
        "FCH4 R2": float(r2_score(y_ch4, p_ch4)),
        "F1 macro": float(f1_score(y_cls, (proba >= 0.5).astype(int), average="macro")),
    }
    out["AUC"] = float(roc_auc_score(y_cls, proba)) if len(np.unique(y_cls)) == 2 else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=14)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    df, feats = D.build_dataset()
    fr = D.train_test_frames(df, feats)
    bundle = joblib.load(MODEL)
    Xte = fr["X_test"]
    p_nee = bundle["regressor_nee"].predict(Xte)
    p_ch4 = bundle["regressor_fch4"].predict(Xte)
    proba = bundle["classifier"].predict_proba(Xte)[:, 1]
    y_nee, y_ch4 = fr["nee_test"].to_numpy(), fr["fch4_test"].to_numpy()
    y_cls = fr["cls_test"].to_numpy()

    point = metrics(y_nee, p_nee, y_ch4, p_ch4, y_cls, proba)
    saved = json.loads(META.read_text(encoding="utf-8"))["metrics_holdout"]
    check = {"NEE RMSE": "nee_rmse", "NEE R2": "nee_r2", "FCH4 RMSE": "fch4_rmse",
             "FCH4 R2": "fch4_r2", "F1 macro": "f1_macro", "AUC": "auc"}
    for k, v in check.items():
        assert abs(point[k] - saved[v]) < 1e-6, f"{k}: {point[k]} != {saved[v]}"
    print("Metricas recalculadas == metadata guardada (modelo_final).")

    rng = np.random.default_rng(C.SEED)
    n = len(y_nee)
    boots = {k: [] for k in point}
    for _ in range(args.reps):
        i = block_indices(n, args.block, rng)
        m = metrics(y_nee[i], p_nee[i], y_ch4[i], p_ch4[i], y_cls[i], proba[i])
        for k, v in m.items():
            boots[k].append(v)

    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        rows.append({"metrica": k, "valor": v, "ic95_inf": lo, "ic95_sup": hi})
    out = pd.DataFrame(rows)
    out["bloque_dias"] = args.block
    out["repeticiones"] = args.reps
    print(out.round(4).to_string(index=False))
    out.to_csv(C.TRAIN_DIR / "metricas_modelo_desplegado.csv", index=False)


if __name__ == "__main__":
    main()
