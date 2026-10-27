"""Sensibilidad del Stacking a la validacion interna que genera las metacaracteristicas.

Responde a la observacion de la revision por pares sobre posible fuga temporal:
el Stacking base usa KFold(5) en bloques sin barajar; aqui se compara con una
variante *purgada*, donde cada bloque de test deja fuera del entrenamiento los
`gap` dias que lo rodean (cortando la autocorrelacion de los retardos y de las
medias moviles). Los bloques de test siguen formando una particion, requisito de
`cross_val_predict`. El holdout 2018 no interviene en ninguna decision.

    python -m ml.training.sensibilidad_stacking --gaps 0 7 30
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from ml.training import config as C
from ml.training import data as D
from ml.training import metrics as M
from ml.training import models_classical as MC

REG_TARGETS = [("nee", C.TARGET_NEE), ("fch4", C.TARGET_FCH4)]


class PurgedBlockKFold:
    """KFold en bloques contiguos; el train excluye `gap` dias a cada lado del test."""

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits, self.gap = n_splits, gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        for tr, te in KFold(self.n_splits, shuffle=False).split(np.arange(n)):
            lo, hi = te.min() - self.gap, te.max() + self.gap
            yield tr[(tr < lo) | (tr > hi)], te


def _stack_reg(cv):
    s = MC.stacking_regressor()
    return StackingRegressor(estimators=s.estimators, final_estimator=LinearRegression(),
                             cv=cv, n_jobs=1)


def _stack_clf(cv, spw):
    s = MC.stacking_classifier(spw)
    return StackingClassifier(estimators=s.estimators,
                              final_estimator=LogisticRegression(class_weight="balanced",
                                                                 max_iter=2000),
                              stack_method="predict_proba", cv=cv, n_jobs=1)


def run(gap: int, fr: dict, spw: float) -> dict:
    cv = KFold(5, shuffle=False) if gap == 0 else PurgedBlockKFold(5, gap)
    row = {"gap_dias": gap}
    for key, _ in REG_TARGETS:
        est = _stack_reg(cv).fit(fr["X_train"], fr[f"{key}_train"])
        m = M.regression_metrics(fr[f"{key}_test"], est.predict(fr["X_test"]))
        row.update({f"{key}_{k}": m[k] for k in ("r2", "rmse")})
    try:
        clf = _stack_clf(cv, spw).fit(fr["X_train"], fr["cls_train"])
        proba = clf.predict_proba(fr["X_test"])[:, 1]
        m = M.classification_metrics(fr["cls_test"], (proba >= 0.5).astype(int), proba)
        row.update({"f1_macro": m["f1_macro"], "auc": m["auc"]})
    except ValueError as e:  # un bloque sin clase 'sumidero'
        row.update({"f1_macro": np.nan, "auc": np.nan, "nota_clf": str(e)[:60]})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=int, nargs="+", default=[0, 7, 30])
    args = ap.parse_args()

    df, feats = D.build_dataset()
    fr = D.train_test_frames(df, feats)
    spw = float((fr["cls_train"] == 0).sum() / (fr["cls_train"] == 1).sum())
    out = pd.DataFrame([run(g, fr, spw) for g in args.gaps])
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv(C.TRAIN_DIR / "sensibilidad_stacking.csv", index=False)


if __name__ == "__main__":
    main()
