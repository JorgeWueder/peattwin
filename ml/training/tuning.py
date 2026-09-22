"""Ajuste de hiperparametros de los modelos mas estables de la CV.

Candidatos (los mejores en cross_validation.py: menor std de R2(FCH4) y mejor
F1_macro): **Random Forest** y **Stacking Ensemble**; **XGBoost** como tercer
candidato opcional.

Estrategia de validacion interna del tuning: **TimeSeriesSplit(5)** de
scikit-learn (NUNCA KFold aleatorio -> fuga de datos en series temporales). El
tuning se hace SOLO sobre el tramo de train (2016-2017); el holdout 2018 queda
intacto para la comparacion final.

  - Regresores: se ajustan envueltos en MultiOutputRegressor sobre [NEE, FCH4] a
    la vez, con scoring = "r2" (media de los 2 R2 -> equivale a reg_score).
  - Clasificadores: scoring = una funcion propia  0.5*F1_macro + 0.5*AUC
    (= cls_score del Paso 5), que ignora la accuracy por el desbalance 32/68.

Criterio de seleccion global (identico al Paso 5):
    combined = 0.5 * reg_score_norm + 0.5 * cls_score
    reg_score_norm = R2 medio(NEE, FCH4) normalizado min-max entre TODOS los
                     modelos del pool (5 base del Paso 5 + versiones tuneadas)
    cls_score      = 0.5 * F1_macro + 0.5 * AUC
Se compara cada modelo tuneado contra su version base y contra el campeon actual
(Stacking Ensemble). Si un modelo tuneado supera el `combined` del campeon base,
se actualiza `modelo_final.pkl`, `modelo_final_metadata.json` y el registro en
`ml_models` (nueva version, is_active = true).

Ejecutar desde la raiz del repositorio:
    python -m ml.training.tuning
    python -m ml.training.tuning --models rf,stacking          # sin xgboost
    python -m ml.training.tuning --fast --no-db                # prueba rapida
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.multioutput import MultiOutputRegressor

from ml.training import config as C
from ml.training import data as D
from ml.training import metrics as M
from ml.training import models_classical as MC
from ml import _logging

# ============================================================ espacios de busqueda
# --- Random Forest (regresion: n_estimators, max_depth, min_samples_leaf) ---
RF_REG_GRID = {
    "estimator__n_estimators": [200, 400, 800],
    "estimator__max_depth": [None, 12, 24],
    "estimator__min_samples_leaf": [1, 2, 4],
}
RF_CLF_GRID = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 12, 24],
    "min_samples_leaf": [1, 2, 4],
}

# --- Stacking: meta-modelo (LinearRegression vs Ridge) + params de los 3 base ---
# El CV INTERNO del stacking de regresion (genera las meta-features) NO se toca:
# usa el mismo KFold(5) en bloques que el modelo base del Paso 5, para que la
# comparacion base-vs-tuneado sea justa. El CV EXTERNO del tuning es
# TimeSeriesSplit(5) (ver _tscv()).
STACK_REG_DIST = {
    "estimator__final_estimator": [LinearRegression(), Ridge(alpha=1.0), Ridge(alpha=10.0)],
    "estimator__rf__n_estimators": [150, 300],
    "estimator__rf__max_depth": [None, 12],
    "estimator__xgb__max_depth": [3, 5],
    "estimator__xgb__learning_rate": [0.03, 0.1],
    "estimator__svr__regressor__svr__C": [1.0, 10.0],
}
# Para clasificacion el analogo de "LinearRegression vs Ridge" es LogisticRegression
# con L2 (= ridge) mas o menos fuerte (parametro C). RidgeClassifier se excluye
# porque NO tiene predict_proba y sin probabilidades no hay AUC.
STACK_CLF_DIST = {
    "final_estimator": [
        LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000),
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000),
        LogisticRegression(class_weight="balanced", C=10.0, max_iter=2000),
    ],
    # CV interno del StackingClassifier: StratifiedKFold(5) SIN barajar (mismo nº
    # de folds que el modelo base del Paso 5, que usaba KFold(5)). Se usa
    # estratificado, no KFold puro, porque la clase 'sumidero' esta concentrada en
    # verano: con folds temporales pequenos un bloque puede quedarse sin ninguna
    # muestra 'sumidero' y cross_val_predict(predict_proba) falla. Para un
    # clasificador es la practica estandar y el riesgo de fuga temporal es bajo.
    "cv": [StratifiedKFold(n_splits=5, shuffle=False)],
    "rf__n_estimators": [150, 300],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.03, 0.1],
    "svc__svc__C": [1.0, 10.0],
}

# --- XGBoost (opcional) ---
XGB_REG_DIST = {
    "estimator__n_estimators": [300, 500, 800],
    "estimator__max_depth": [3, 4, 6],
    "estimator__learning_rate": [0.03, 0.05, 0.1],
    "estimator__subsample": [0.8, 1.0],
    "estimator__colsample_bytree": [0.8, 1.0],
}
XGB_CLF_DIST = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

REG_TARGETS = ["NEE_F_ANNOPTLM", "FCH4_F_ANNOPTLM"]


def cls_scorer(estimator, X, y):
    """0.5*F1_macro + 0.5*AUC  (= cls_score del Paso 5)."""
    proba = estimator.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return 0.5 * f1_score(y, pred, average="macro") + 0.5 * roc_auc_score(y, proba)


def _tscv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=C.TUNING_TSCV_SPLITS)


def _search(kind, estimator, space, X, y, n_iter, seed=C.SEED):
    common = dict(cv=_tscv(), n_jobs=-1, refit=True, error_score=np.nan)
    if kind == "grid":
        gs = GridSearchCV(estimator, space, scoring="r2" if _is_reg(y) else cls_scorer, **common)
    else:
        gs = RandomizedSearchCV(estimator, space, n_iter=n_iter, random_state=seed,
                                scoring="r2" if _is_reg(y) else cls_scorer, **common)
    t0 = time.perf_counter()
    gs.fit(X, y)
    return gs, time.perf_counter() - t0


def _is_reg(y) -> bool:
    y = np.asarray(y)
    return y.ndim == 2 or np.unique(y).size > 2


# ==================================================================== tuners
def tune_random_forest(fr: dict, fast: bool) -> dict:
    grid = _shrink(RF_REG_GRID, fast)
    mor = MultiOutputRegressor(MC.rf_regressor().set_params(n_jobs=1))
    yreg = fr["df"].loc[fr["tr"], REG_TARGETS].to_numpy()
    gs_r, t_r = _search("grid", mor, grid, fr["X_train"], yreg, None)

    gs_c, t_c = _search("grid", MC.rf_classifier().set_params(n_jobs=1),
                        _shrink(RF_CLF_GRID, fast), fr["X_train"], fr["cls_train"], None)
    return _pack("Random Forest (tuned)", "CLASSICAL", gs_r, gs_c, t_r + t_c, fr,
                 {"regresion": _clean(gs_r.best_params_), "clasificacion": _clean(gs_c.best_params_)})


def tune_stacking(fr: dict, fast: bool) -> dict:
    spw = fr["spw"]
    mor = MultiOutputRegressor(MC.stacking_regressor())
    yreg = fr["df"].loc[fr["tr"], REG_TARGETS].to_numpy()
    gs_r, t_r = _search("random", mor, _shrink(STACK_REG_DIST, fast),
                        fr["X_train"], yreg, 3 if fast else 8)

    gs_c, t_c = _search("random", MC.stacking_classifier(spw), _shrink(STACK_CLF_DIST, fast),
                        fr["X_train"], fr["cls_train"], 3 if fast else 8)
    return _pack("Stacking Ensemble (tuned)", "HYBRID", gs_r, gs_c, t_r + t_c, fr,
                 {"regresion": _clean(gs_r.best_params_), "clasificacion": _clean(gs_c.best_params_)})


def tune_xgboost(fr: dict, fast: bool) -> dict:
    spw = fr["spw"]
    mor = MultiOutputRegressor(MC.xgb_regressor().set_params(n_jobs=1))
    yreg = fr["df"].loc[fr["tr"], REG_TARGETS].to_numpy()
    gs_r, t_r = _search("random", mor, _shrink(XGB_REG_DIST, fast),
                        fr["X_train"], yreg, 3 if fast else 15)

    gs_c, t_c = _search("random", MC.xgb_classifier(spw).set_params(n_jobs=1),
                        _shrink(XGB_CLF_DIST, fast), fr["X_train"], fr["cls_train"],
                        3 if fast else 15)
    return _pack("Gradient Boosting (XGBoost) (tuned)", "CLASSICAL", gs_r, gs_c, t_r + t_c, fr,
                 {"regresion": _clean(gs_r.best_params_), "clasificacion": _clean(gs_c.best_params_)})


# ---------------------------------------------------------------------- helpers
def _shrink(space: dict, fast: bool) -> dict:
    if not fast:
        return space
    return {k: v[:2] for k, v in space.items()}


def _estimator_label(v) -> str:
    name = v.__class__.__name__
    if hasattr(v, "alpha"):
        return f"{name}(alpha={v.alpha})"
    if hasattr(v, "C"):
        return f"{name}(C={v.C})"
    return name


def _clean(params: dict) -> dict:
    """Convierte los best_params_ a algo JSON-serializable (estimadores, splitters...)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif hasattr(v, "get_params"):
            out[k] = _estimator_label(v)
        else:
            out[k] = repr(v)  # KFold(...), StratifiedKFold(...), etc.
    return out


def _pack(name, algo, gs_r, gs_c, t_train, fr, best_params) -> dict:
    """Evalua el modelo tuneado en el holdout 2018 y devuelve la fila de resultados."""
    reg = gs_r.best_estimator_           # MultiOutputRegressor ya reentrenado en train
    clf = gs_c.best_estimator_
    pred = reg.predict(fr["X_test"])
    row = {"model": name, "algorithm_type": algo, "train_time_s": round(t_train, 2),
           "cv_score_reg": float(gs_r.best_score_), "cv_score_cls": float(gs_c.best_score_)}
    for j, col in enumerate(REG_TARGETS):
        key = "nee" if j == 0 else "fch4"
        m = M.regression_metrics(fr["df"].loc[fr["te"], col], pred[:, j])
        row.update({f"{key}_{k}": v for k, v in m.items()})
    proba = clf.predict_proba(fr["X_test"])[:, 1]
    row.update(M.classification_metrics(fr["cls_test"], (proba >= 0.5).astype(int), proba))
    row["_estimators"] = {"regressor_nee": reg.estimators_[0],
                          "regressor_fch4": reg.estimators_[1], "classifier": clf}
    row["_best_params"] = best_params
    return row


# ============================================================ criterio combinado
def build_pool(tuned_rows: list[dict]) -> pd.DataFrame:
    base = pd.read_csv(C.RESULTS_CSV)
    cols = ["model", "algorithm_type", "nee_r2", "fch4_r2", "f1_macro", "auc",
            "nee_rmse", "fch4_rmse", "recall_sumidero", "accuracy", "train_time_s"]
    pool = base[cols].copy()
    for r in tuned_rows:
        pool.loc[len(pool)] = {c: r.get(c, np.nan) for c in cols}

    r2_mean = pool[["nee_r2", "fch4_r2"]].mean(axis=1).to_numpy()
    pool["reg_score_norm"] = M.minmax_norm(r2_mean)
    pool["cls_score"] = 0.5 * pool["f1_macro"] + 0.5 * pool["auc"]
    pool["combined_score"] = 0.5 * pool["reg_score_norm"] + 0.5 * pool["cls_score"]
    return pool


# ================================================================ persistencia
def update_final_model(win_row: dict, pool: pd.DataFrame) -> str:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    est = win_row["_estimators"]
    bundle = {
        "model_name": win_row["model"],
        "regressor_nee": est["regressor_nee"],
        "regressor_fch4": est["regressor_fch4"],
        "classifier": est["classifier"],
        "positive_class": "sumidero (1)",
        "tuned": True,
        "best_params": win_row["_best_params"],
        "note": "Version ajustada (tuning.py). .h5 no aplica: es scikit-learn -> joblib.",
    }
    joblib.dump(bundle, C.FINAL_PKL)

    meta = {
        "winner": win_row["model"],
        "tuned": True,
        "criterio": "0.5*reg_score_norm + 0.5*(0.5*F1_macro + 0.5*AUC)",
        "combined_score": float(pool.loc[pool["model"] == win_row["model"], "combined_score"].iloc[0]),
        "best_params": win_row["_best_params"],
        "saved_as": str(C.FINAL_PKL), "format": "pkl",
        "holdout": {"train_years": [2016, 2017], "test_year": 2018},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_holdout": {k: v for k, v in win_row.items()
                            if not k.startswith("_") and k != "model"},
    }
    C.FINAL_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(C.FINAL_PKL)


def register_in_db(win_row: dict, pool: pd.DataFrame, model_path: str) -> None:
    from sqlalchemy import MetaData, Table, create_engine, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from ml.etl.load_to_db import _database_url

    engine = create_engine(_database_url(None), future=True)
    md = MetaData()
    ml_models = Table("ml_models", md, autoload_with=engine)
    ml_metrics = Table("ml_model_metrics", md, autoload_with=engine)
    base_name = win_row["model"]
    algo = win_row["algorithm_type"]
    now = datetime.now(timezone.utc)
    prow = pool.loc[pool["model"] == base_name].iloc[0]

    def upsert_model(conn, name, task, target, desc):
        stmt = pg_insert(ml_models).values(
            name=name, version="1.1-tuned", algorithm_type=algo, task=task,
            file_path=model_path, framework="scikit-learn", target_variable=target,
            trained_at=now, is_active=True, description=desc,
        )
        return conn.execute(stmt.on_conflict_do_update(
            constraint="uq_ml_model_name_version",
            set_={"file_path": model_path, "trained_at": now, "is_active": True,
                  "description": desc},
        ).returning(ml_models.c.id)).scalar_one()

    def upsert_metric(conn, mid, **vals):
        vals = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in vals.items()}
        stmt = pg_insert(ml_metrics).values(model_id=mid, fold=-1, **vals)
        conn.execute(stmt.on_conflict_do_update(
            constraint="uq_ml_model_metric_model_fold", set_=vals))

    with engine.begin() as conn:
        conn.execute(ml_models.update()
                     .where(ml_models.c.task.in_(["REGRESSION", "CLASSIFICATION"]))
                     .values(is_active=False))
        reg_id = upsert_model(
            conn, f"{base_name} - regresion", "REGRESSION", "NEE_F_ANNOPTLM+FCH4_F_ANNOPTLM",
            f"Tuned. R2 NEE={win_row['nee_r2']:.3f}, R2 FCH4={win_row['fch4_r2']:.3f}. "
            f"best_params: {win_row['_best_params']['regresion']}")
        upsert_metric(conn, reg_id,
                      rmse=float(np.mean([win_row["nee_rmse"], win_row["fch4_rmse"]])),
                      mae=float(np.mean([win_row["nee_mae"], win_row["fch4_mae"]])),
                      r2=float(np.mean([win_row["nee_r2"], win_row["fch4_r2"]])),
                      nse=float(np.nanmean([win_row["nee_nse"], win_row["fch4_nse"]])),
                      kge=float(np.nanmean([win_row["nee_kge"], win_row["fch4_kge"]])),
                      training_time_seconds=float(win_row["train_time_s"]))
        cls_id = upsert_model(
            conn, f"{base_name} - clasificacion", "CLASSIFICATION", "clase_balance_carbono",
            f"Tuned. F1_macro={win_row['f1_macro']:.3f}, AUC={win_row['auc']:.3f}. "
            f"best_params: {win_row['_best_params']['clasificacion']}")
        upsert_metric(conn, cls_id, accuracy=float(win_row["accuracy"]),
                      precision=float(win_row["precision_macro"]),
                      recall=float(win_row["recall_macro"]), f1=float(win_row["f1_macro"]),
                      auc_roc=float(win_row["auc"]), training_time_seconds=float(win_row["train_time_s"]))
        active = conn.execute(select(ml_models.c.name, ml_models.c.task, ml_models.c.version)
                              .where(ml_models.c.is_active.is_(True))).all()
    print("  ml_models activos:", [(a.name, a.task, a.version) for a in active])


def write_summary_md(pool: pd.DataFrame, tuned_rows: list[dict], decision: str) -> None:
    spaces = {
        "Random Forest - regresion": RF_REG_GRID,
        "Random Forest - clasificacion": RF_CLF_GRID,
        "Stacking - regresion": STACK_REG_DIST,
        "Stacking - clasificacion": STACK_CLF_DIST,
        "XGBoost - regresion": XGB_REG_DIST,
        "XGBoost - clasificacion": XGB_CLF_DIST,
    }
    sp_md = "\n".join(
        f"### {name}\n\n" + "\n".join(f"- `{k}`: {_space_str(v)}" for k, v in grid.items())
        for name, grid in spaces.items()
    )
    show = pool[["model", "nee_r2", "fch4_r2", "f1_macro", "auc", "recall_sumidero",
                 "reg_score_norm", "cls_score", "combined_score"]].round(4)
    tbl = "| " + " | ".join(show.columns) + " |\n| " + " | ".join(["---"] * len(show.columns)) + " |\n"
    tbl += "\n".join("| " + " | ".join(str(x) for x in row) + " |"
                     for row in show.itertuples(index=False, name=None))
    bp = "\n".join(f"### {r['model']}\n```\n{json.dumps(r['_best_params'], indent=2, ensure_ascii=False)}\n```"
                   for r in tuned_rows)

    md = f"""# Ajuste de hiperparametros (tuning)

Candidatos: **Random Forest** y **Stacking Ensemble** (los mas estables en la
validacion cruzada) + **XGBoost** (opcional). Busqueda con GridSearchCV /
RandomizedSearchCV y **TimeSeriesSplit({C.TUNING_TSCV_SPLITS})** como CV interna
(nunca KFold aleatorio). Tuning solo sobre train 2016-2017; holdout 2018 intacto.

## Espacio de busqueda

{sp_md}

**Notas:**
- Los regresores se ajustan con `MultiOutputRegressor` sobre `[NEE, FCH4]` y
  `scoring="r2"` (media de los dos R2 = `reg_score`).
- Los clasificadores usan un scorer propio `0.5*F1_macro + 0.5*AUC` (= `cls_score`),
  que ignora la accuracy por el desbalance 32/68.
- El CV **interno** del stacking (genera las meta-features): `KFold(5)` en
  bloques para la regresion (igual que el modelo base del Paso 5, para una
  comparacion justa) y `StratifiedKFold(5)` sin barajar para la clasificacion
  (estratificado, no KFold puro, porque 'sumidero' se concentra en verano y un
  bloque temporal pequeno podria quedarse sin esa clase, rompiendo
  `cross_val_predict(predict_proba)`). `TimeSeriesSplit` no sirve como CV interno
  porque `cross_val_predict` exige una particion completa. El CV **externo** del
  tuning si es `TimeSeriesSplit(5)`.
- El meta-learner del stacking de clasificacion se prueba como `LogisticRegression`
  con L2 (= ridge) a distinta fuerza (`C`); `RidgeClassifier` se excluye por no
  tener `predict_proba` (sin probabilidades no hay AUC).

## Criterio de seleccion

`combined = 0.5 * reg_score_norm + 0.5 * cls_score` (identico al Paso 5).
`reg_score_norm` = R2 medio(NEE, FCH4) normalizado min-max sobre el pool completo
(5 modelos base + versiones tuneadas). `cls_score = 0.5*F1_macro + 0.5*AUC`.

## Mejores hiperparametros encontrados

{bp}

## Comparativa base vs tuneado (holdout 2018)

`reg_score_norm` y `combined_score` se **recalculan sobre el pool ampliado**
(5 modelos base + tuneados), por lo que los valores absolutos difieren de los del
Paso 5 (que tenia 5 modelos). La comparacion base-vs-tuneado de esta tabla si es
homogenea (mismo pool).

{tbl}

## Decision

{decision}

*(generado por `python -m ml.training.tuning`)*
"""
    C.TUNING_MD.write_text(md, encoding="utf-8")
    print(f"  resumen -> {C.TUNING_MD.relative_to(C.ROOT)}")


def _space_str(v) -> str:
    return "[" + ", ".join(
        (_estimator_label(x) if hasattr(x, "get_params") else str(x)) for x in v
    ) + "]"


# ======================================================================= main
def main() -> None:
    _logging.setup("tuning")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="rf,stacking,xgboost",
                    help="cuales ajustar (coma): rf, stacking, xgboost")
    ap.add_argument("--fast", action="store_true", help="grids reducidos (prueba rapida)")
    ap.add_argument("--no-db", action="store_true", help="no actualizar ml_models")
    args = ap.parse_args()
    which = {m.strip() for m in args.models.split(",")}

    print("=" * 79)
    print(f"Tuning de hiperparametros  (TimeSeriesSplit={C.TUNING_TSCV_SPLITS}, "
          f"modelos={sorted(which)}, fast={args.fast})")
    print("=" * 79)

    df, feats = D.build_dataset()
    base_fr = D.train_test_frames(df, feats)
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()
    fr = {**base_fr, "df": df, "tr": tr, "te": te,
          "spw": float((base_fr["cls_train"] == 0).sum() / (base_fr["cls_train"] == 1).sum())}
    print(f"train={tr.sum()}  test={te.sum()}  features={len(feats)}\n")

    tuners = []
    if "rf" in which:
        tuners.append(("Random Forest", tune_random_forest))
    if "stacking" in which:
        tuners.append(("Stacking Ensemble", tune_stacking))
    if "xgboost" in which:
        tuners.append(("Gradient Boosting (XGBoost)", tune_xgboost))

    tuned_rows = []
    for label, fn in tuners:
        print(f"-- ajustando {label} ...")
        t0 = time.perf_counter()
        row = fn(fr, args.fast)
        tuned_rows.append(row)
        print(f"   listo en {time.perf_counter() - t0:.0f}s  "
              f"holdout: R2(NEE)={row['nee_r2']:+.3f} R2(FCH4)={row['fch4_r2']:+.3f} "
              f"F1_macro={row['f1_macro']:.3f} AUC={row['auc']:.3f}")
        print(f"   best regresion   : {row['_best_params']['regresion']}")
        print(f"   best clasificacion: {row['_best_params']['clasificacion']}")

    pool = build_pool(tuned_rows)
    pool.to_csv(C.TUNING_CSV, index=False)
    print("-" * 79)
    print(pool[["model", "nee_r2", "fch4_r2", "f1_macro", "auc",
                "reg_score_norm", "cls_score", "combined_score"]].round(4).to_string(index=False))

    # ¿mejora el campeon actual (Stacking Ensemble base)?
    champ_base = float(pool.loc[pool["model"] == "Stacking Ensemble", "combined_score"].iloc[0])
    tuned_mask = pool["model"].str.contains(r"\(tuned\)", regex=True)
    best_tuned = pool[tuned_mask].sort_values("combined_score").iloc[-1]
    best_tuned_row = next(r for r in tuned_rows if r["model"] == best_tuned["model"])
    improves = bool(best_tuned["combined_score"] > champ_base)

    print("-" * 79)
    print(f"campeon base (Stacking Ensemble): combined = {champ_base:.4f}")
    print(f"mejor tuneado ({best_tuned['model']}):      combined = {best_tuned['combined_score']:.4f}")

    if improves:
        path = update_final_model(best_tuned_row, pool)
        decision = (f"**{best_tuned['model']}** mejora el campeon base "
                    f"({best_tuned['combined_score']:.4f} > {champ_base:.4f}). "
                    f"Se actualiza `modelo_final.pkl` y el registro en `ml_models` "
                    f"(version 1.1-tuned, is_active=true).")
        print(f"  -> modelo_final.pkl ACTUALIZADO ({best_tuned['model']})")
        if not args.no_db:
            try:
                register_in_db(best_tuned_row, pool, path)
            except Exception as exc:  # pragma: no cover
                print(f"  AVISO: no se pudo registrar en la BD: {exc}")
    else:
        decision = (f"Ningun modelo tuneado supera el `combined` del campeon base "
                    f"({best_tuned['combined_score']:.4f} <= {champ_base:.4f}). "
                    f"Se conserva `modelo_final.pkl` del Paso 5 sin cambios.")
        print("  -> sin cambios en modelo_final.pkl")

    write_summary_md(pool, tuned_rows, decision)
    print("-" * 79)
    print(f"tabla -> {C.TUNING_CSV.relative_to(C.ROOT)}")
    print("OK.")


if __name__ == "__main__":
    main()
