"""Contrastes estadisticos entre los modelos AJUSTADOS (Random Forest, Stacking, XGBoost).

Complementa `statistical_tests.py` (que solo cubre los 5 modelos base): el modelo
desplegado es `Stacking Ensemble (tuned)`, asi que aqui se repite la validacion
cruzada temporal (mismos 5 folds, sin separacion) y la prueba de Diebold-Mariano del
holdout 2018 con los hiperparametros hallados por `tuning.py` (leidos de
TUNING_resumen.md). Reutiliza `holm` y `diebold_mariano` del modulo base.

    python -m ml.training.statistical_tests_tuned
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from ml.training import config as C
from ml.training import data as D
from ml.training import metrics as M
from ml.training import models_classical as MC
from ml.training.statistical_tests import diebold_mariano, holm

NAMES = ["Random Forest (tuned)", "Stacking Ensemble (tuned)",
         "Gradient Boosting (XGBoost) (tuned)"]
REFERENCE = "Stacking Ensemble (tuned)"
REG = [("nee", C.TARGET_NEE), ("fch4", C.TARGET_FCH4)]


# ------------------------------------------------------------ hiperparametros
def best_params() -> dict:
    txt = C.TUNING_MD.read_text(encoding="utf-8")
    out = {}
    for name in NAMES:
        m = re.search(rf"### {re.escape(name)}\n```\n(.*?)\n```", txt, re.S)
        out[name] = json.loads(m.group(1))
    return out


def _final(s: str):
    if s.startswith("LinearRegression"):
        return LinearRegression()
    if s.startswith("Ridge"):
        return Ridge(alpha=float(re.search(r"alpha=([\d.]+)", s).group(1)))
    if s.startswith("LogisticRegression"):
        return LogisticRegression(class_weight="balanced", max_iter=2000,
                                  C=float(re.search(r"C=([\d.]+)", s).group(1)))
    raise ValueError(s)


def _clean(params: dict, strip: str) -> dict:
    out = {}
    for k, v in params.items():
        k = k.removeprefix(strip)
        if k == "cv":
            v = StratifiedKFold(n_splits=5, shuffle=False)
        elif k == "final_estimator":
            v = _final(v)
        out[k] = v
    return out


def builders(name: str, params: dict, spw: float):
    reg_p = _clean(params["regresion"], "estimator__")
    clf_p = _clean(params["clasificacion"], "")
    if name.startswith("Random Forest"):
        return (lambda: MC.rf_regressor().set_params(**reg_p),
                lambda: MC.rf_classifier().set_params(**clf_p))
    if name.startswith("Stacking"):
        return (lambda: MC.stacking_regressor().set_params(**reg_p),
                lambda: MC.stacking_classifier(spw).set_params(**clf_p))
    return (lambda: MC.xgb_regressor().set_params(**reg_p),
            lambda: MC.xgb_classifier(spw).set_params(**clf_p))


def evaluate(make_reg, make_clf, df, feats, tr, te) -> tuple[float, float]:
    Xtr, Xte = df[feats].iloc[tr], df[feats].iloc[te]
    rm = []
    for _, col in REG:
        est = make_reg().fit(Xtr, df[col].iloc[tr])
        rm.append(M.regression_metrics(df[col].iloc[te], est.predict(Xte))["rmse"])
    ctr = df[C.TARGET_CLS].iloc[tr].astype(int)
    cte = df[C.TARGET_CLS].iloc[te].astype(int)
    clf = make_clf().fit(Xtr, ctr)
    proba = clf.predict_proba(Xte)[:, 1]
    f1 = M.classification_metrics(cte, (proba >= 0.5).astype(int), proba)["f1_macro"]
    return float(np.mean(rm)), float(f1)


def paired(mat: pd.DataFrame, label: str) -> pd.DataFrame:
    recs, raw = [], []
    for m in NAMES:
        if m == REFERENCE:
            continue
        a, b = mat[REFERENCE].to_numpy(), mat[m].to_numpy()
        try:
            st, p = stats.wilcoxon(a, b)
        except ValueError:
            st, p = np.nan, 1.0
        recs.append({"comparacion": f"{REFERENCE} vs {m}", "metrica": label,
                     "test": "Wilcoxon signed-rank", "p_value": p,
                     "mean_diff": float(np.mean(a - b))})
        raw.append(p)
    for r, ph in zip(recs, holm(raw)):
        r["p_holm"] = ph
    return pd.DataFrame(recs)


def main() -> None:
    df, feats = D.build_dataset()
    params = best_params()
    n = len(df)
    folds = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=C.CV_N_SPLITS).split(np.arange(n))):
        ctr = df[C.TARGET_CLS].iloc[tr].astype(int)
        spw = float((ctr == 0).sum() / max((ctr == 1).sum(), 1))
        for name in NAMES:
            mk_r, mk_c = builders(name, params[name], spw)
            rmse, f1 = evaluate(mk_r, mk_c, df, feats, tr, te)
            folds.append({"model": name, "fold": fold, "rmse": rmse, "f1": f1})
            print(f"fold {fold}  {name:38s} RMSE={rmse:8.3f}  F1={f1:.3f}")
    long = pd.DataFrame(folds)
    long.to_csv(C.TRAIN_DIR / "cv_por_fold_ajustados.csv", index=False)
    reg = long.pivot(index="fold", columns="model", values="rmse")[NAMES]
    cls = long.pivot(index="fold", columns="model", values="f1")[NAMES]

    rows = []
    for mat, lab in ((reg, "RMSE (CV)"), (cls, "F1_macro (CV)")):
        pr = paired(mat, lab)
        st, p = stats.friedmanchisquare(*[mat[m] for m in NAMES])
        print(f"\nFriedman {lab}: chi2={st:.3f} p={p:.4f}")
        print(pr[["comparacion", "p_value", "p_holm", "mean_diff"]].round(4).to_string(index=False))
        rows += pr.to_dict("records")
        rows.append({"comparacion": "Friedman (3 modelos)", "metrica": lab,
                     "test": "Friedman", "p_value": p, "statistic": st})

    # Diebold-Mariano en el holdout 2018 (Stacking ajustado vs los otros dos ajustados)
    fr = D.train_test_frames(df, feats)
    ctr = fr["cls_train"]
    spw = float((ctr == 0).sum() / (ctr == 1).sum())
    errs = {}
    for name in NAMES:
        mk_r, _ = builders(name, params[name], spw)
        for key, _ in REG:
            est = mk_r().fit(fr["X_train"], fr[f"{key}_train"])
            errs[(name, key)] = fr[f"{key}_test"].to_numpy() - est.predict(fr["X_test"])
    print("\nDiebold-Mariano (holdout 2018, perdida cuadratica):")
    for name in NAMES:
        if name == REFERENCE:
            continue
        for key, _ in REG:
            r = diebold_mariano(errs[(REFERENCE, key)], errs[(name, key)])
            print(f"  {key.upper():5s} {REFERENCE} vs {name}: DM={r['DM_stat']:+.3f} "
                  f"p={r['p_value']:.3f} lag={r['lag_NW']} n={r['n']}")
            rows.append({"comparacion": f"DM {key.upper()}: {REFERENCE} vs {name}",
                         "metrica": "holdout 2018", "test": "Diebold-Mariano",
                         "statistic": r["DM_stat"], "p_value": r["p_value"],
                         "mean_diff": r["mean_loss_diff"]})
    pd.DataFrame(rows).to_csv(C.TRAIN_DIR / "statistical_tests_ajustados_resultados.csv",
                              index=False)


if __name__ == "__main__":
    main()
