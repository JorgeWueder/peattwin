"""Carga de datos y utilidades compartidas por los tres generadores de informe."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import DomainError
from app.models.flux_observation import FluxObservation
from app.models.ml_model import MLModel
from app.models.ml_model_metric import MLModelMetric
from app.models.prediction import Prediction
from app.models.restoration_scenario import RestorationScenario
from app.models.site import Site

# --------------------------------------------------------------- rutas artefactos
TRAIN_DIR = settings.project_root / "ml" / "training"
TRAIN_FIG = TRAIN_DIR / "figuras"
EDA_DIR = settings.project_root / "ml" / "eda"
EDA_FIG = EDA_DIR / "figuras"
EDA_TBL = EDA_DIR / "tablas"
PROCESSED = settings.processed_dir


def fig(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def fnum(v, digits: int = 3, dash: str = "—") -> str:
    try:
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            return dash
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------- DB helpers
def get_site(db: Session, site_id: int) -> Site:
    site = db.get(Site, site_id)
    if site is None:
        raise DomainError("Sitio no encontrado", "not_found")
    return site


def flux_rows(db: Session, site_id: int, start: date | None, end: date | None) -> list[FluxObservation]:
    stmt = select(FluxObservation).where(FluxObservation.site_id == site_id).order_by(
        FluxObservation.timestamp
    )
    if start:
        stmt = stmt.where(FluxObservation.timestamp >= datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
    if end:
        stmt = stmt.where(FluxObservation.timestamp <= datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc))
    return list(db.scalars(stmt))


def prediction_rows(db: Session, site_id: int) -> list[dict]:
    stmt = (
        select(Prediction, RestorationScenario.name, MLModel.name)
        .join(RestorationScenario, RestorationScenario.id == Prediction.scenario_id)
        .join(MLModel, MLModel.id == Prediction.model_id)
        .where(RestorationScenario.site_id == site_id)
        .order_by(Prediction.timestamp)
    )
    out = []
    for pred, scen_name, model_name in db.execute(stmt):
        out.append(
            {
                "id": pred.id,
                "escenario": scen_name,
                "modelo": model_name,
                "timestamp": pred.timestamp,
                "co2_predicted": pred.co2_predicted,
                "ch4_predicted": pred.ch4_predicted,
                "predicted_class": pred.predicted_class,
            }
        )
    return out


@dataclass
class ModelRow:
    name: str
    algorithm_type: str
    framework: str | None
    version: str
    is_active: bool
    # regresion (media NEE+FCH4, holdout fold -1)
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    nse: float | None = None
    kge: float | None = None
    train_time_s: float | None = None
    # clasificacion (holdout fold -1)
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc: float | None = None
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    tp: int | None = None


def _base_name(n: str) -> str:
    return n.replace(" - regresion", "").replace(" - clasificacion", "").strip()


def model_rows(db: Session) -> list[ModelRow]:
    models = list(db.scalars(select(MLModel).order_by(MLModel.id)))
    metrics = list(db.scalars(select(MLModelMetric)))
    holdout: dict[int, MLModelMetric] = {m.model_id: m for m in metrics if m.fold == -1}

    merged: dict[str, ModelRow] = {}
    for model in models:
        key = _base_name(model.name)
        row = merged.get(key) or ModelRow(
            name=key,
            algorithm_type=str(getattr(model.algorithm_type, "value", model.algorithm_type)),
            framework=model.framework,
            version=model.version,
            is_active=False,
        )
        row.is_active = row.is_active or model.is_active
        row.version = model.version
        h = holdout.get(model.id)
        task = str(getattr(model.task, "value", model.task))
        if h is not None and task == "REGRESSION":
            row.rmse, row.mae, row.r2 = h.rmse, h.mae, h.r2
            row.nse, row.kge = h.nse, h.kge
            row.train_time_s = h.training_time_seconds
        elif h is not None and task == "CLASSIFICATION":
            row.accuracy, row.precision, row.recall = h.accuracy, h.precision, h.recall
            row.f1, row.auc = h.f1, h.auc_roc
        merged[key] = row

    rows = sorted(merged.values(), key=lambda r: (not r.is_active, r.name.lower()))
    # la matriz de confusion (tn/fp/fn/tp) no esta en ml_model_metrics; se toma
    # del metadata del modelo desplegado para la fila activa.
    wmeta = winner_holdout_meta()
    for r in rows:
        if r.is_active and wmeta:
            r.tn = wmeta.get("tn")
            r.fp = wmeta.get("fp")
            r.fn = wmeta.get("fn")
            r.tp = wmeta.get("tp")
    return rows


def winner_holdout_meta() -> dict | None:
    """metrics_holdout de ml/models/modelo_final_metadata.json (incl. tn/fp/fn/tp)."""
    import json

    path = settings.models_dir / "modelo_final_metadata.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return d.get("metrics_holdout") or d.get("metrics_winner")


def confusion_from_series(hs: "HoldoutSeries", threshold: float = 0.5):
    """(tn, fp, fn, tp) a partir de y_true (1=sumidero) y proba del backtest."""
    tn = fp = fn = tp = 0
    for t, p in zip(hs.y_true, hs.proba):
        pred = 1 if p >= threshold else 0
        if t == 1 and pred == 1:
            tp += 1
        elif t == 1 and pred == 0:
            fn += 1
        elif t == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    return tn, fp, fn, tp


def fold_metric_rows(db: Session) -> list[dict]:
    """Todas las filas de ml_model_metrics con el nombre del modelo (para el Excel)."""
    stmt = (
        select(MLModelMetric, MLModel.name, MLModel.task, MLModel.version)
        .join(MLModel, MLModel.id == MLModelMetric.model_id)
        .order_by(MLModel.name, MLModelMetric.fold)
    )
    out = []
    for mm, name, task, version in db.execute(stmt):
        out.append(
            {
                "modelo": name,
                "tarea": str(getattr(task, "value", task)),
                "version": version,
                "fold": mm.fold,
                "rmse": mm.rmse,
                "mae": mm.mae,
                "r2": mm.r2,
                "nse": mm.nse,
                "kge": mm.kge,
                "accuracy": mm.accuracy,
                "precision": mm.precision,
                "recall": mm.recall,
                "f1": mm.f1,
                "auc_roc": mm.auc_roc,
                "training_time_seconds": mm.training_time_seconds,
            }
        )
    return out


def cv_summary_rows(db: Session) -> list[dict]:
    """Media y desviacion estandar por modelo sobre los folds 0..k-1."""
    metrics = list(db.scalars(select(MLModelMetric)))
    models = {m.id: m for m in db.scalars(select(MLModel))}
    by_base: dict[str, dict[str, list[float]]] = {}
    for mm in metrics:
        if mm.fold < 0:
            continue
        model = models.get(mm.model_id)
        if model is None:
            continue
        base = _base_name(model.name)
        task = str(getattr(model.task, "value", model.task))
        d = by_base.setdefault(base, {"rmse": [], "r2": [], "f1": [], "auc": [], "folds": []})
        d["folds"].append(mm.fold)
        if task == "REGRESSION":
            if mm.rmse is not None:
                d["rmse"].append(mm.rmse)
            if mm.r2 is not None:
                d["r2"].append(mm.r2)
        else:
            if mm.f1 is not None:
                d["f1"].append(mm.f1)
            if mm.auc_roc is not None:
                d["auc"].append(mm.auc_roc)

    import statistics

    def ms(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    out = []
    for base, d in sorted(by_base.items()):
        rm, rs = ms(d["rmse"])
        r2m, r2s = ms(d["r2"])
        f1m, f1s = ms(d["f1"])
        aucm, aucs = ms(d["auc"])
        out.append(
            {
                "modelo": base,
                "n_folds": len(set(d["folds"])),
                "rmse_mean": rm, "rmse_std": rs,
                "r2_mean": r2m, "r2_std": r2s,
                "f1_mean": f1m, "f1_std": f1s,
                "auc_mean": aucm, "auc_std": aucs,
            }
        )
    return out


@dataclass
class HoldoutSeries:
    dates: list = field(default_factory=list)
    nee_obs: list = field(default_factory=list)
    nee_pred: list = field(default_factory=list)
    fch4_obs: list = field(default_factory=list)
    fch4_pred: list = field(default_factory=list)
    y_true: list = field(default_factory=list)   # 1 = sumidero
    proba: list = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""
    available: bool = False


def holdout_series(db: Session, site: Site, start: date | None, end: date | None) -> HoldoutSeries:
    """Ejecuta el modelo activo sobre las features reales del sitio (backtest)."""
    if site.fluxnet_id != "DE-Zrk":
        return HoldoutSeries()
    from app.ml import predictor

    try:
        active = predictor.load_active_model(db)
        df, feats = predictor.feature_frame()
    except FileNotFoundError:
        return HoldoutSeries()

    df = df.copy()
    df["d"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.sort_values("d")
    if start:
        df = df[df["d"] >= start]
    if end:
        df = df[df["d"] <= end]
    if df.empty:
        return HoldoutSeries(model_name=active.name, model_version=active.version, available=False)

    nee_p, fch4_p, proba = predictor.predict_frame(active, df[feats])
    nee_obs = df["NEE_F_ANNOPTLM"].astype(float).tolist()
    fch4_obs = df["FCH4_F_ANNOPTLM"].astype(float).tolist()
    return HoldoutSeries(
        dates=[d.isoformat() for d in df["d"].tolist()],
        nee_obs=nee_obs,
        nee_pred=nee_p,
        fch4_obs=fch4_obs,
        fch4_pred=fch4_p,
        y_true=[1 if v < 0 else 0 for v in nee_obs],
        proba=proba,
        model_name=active.name,
        model_version=active.version,
        available=True,
    )


def regression_scores(obs: list[float], pred: list[float]) -> tuple[float | None, float | None]:
    """(rmse, r2) sobre pares alineados."""
    pairs = [(o, p) for o, p in zip(obs, pred) if o is not None and p is not None]
    if not pairs:
        return None, None
    n = len(pairs)
    rmse = (sum((o - p) ** 2 for o, p in pairs) / n) ** 0.5
    mean = sum(o for o, _ in pairs) / n
    ss_res = sum((o - p) ** 2 for o, p in pairs)
    ss_tot = sum((o - mean) ** 2 for o, _ in pairs)
    r2 = None if ss_tot == 0 else 1 - ss_res / ss_tot
    return rmse, r2


def peatland_label(t) -> str:
    m = {
        "BOG": "bog (ombrotrófica)", "FEN": "fen (minerotrófica)",
        "BLANKET_BOG": "blanket bog", "RAISED_BOG": "raised bog",
        "TRANSITIONAL_MIRE": "turbera de transición", "SWAMP_FOREST": "bosque pantanoso",
        "OTHER": "otro",
    }
    key = str(getattr(t, "value", t)) if t is not None else None
    return m.get(key, key or "—")


def scenario_prediction(db: Session, *, scenario_id: int | None, wtd_cm: float | None, on_date: date):
    """Devuelve un dict con la predicción del simulador para el informe técnico."""
    from app.schemas.predict import PredictRequest
    from app.services import prediction_service

    if scenario_id is not None:
        payload = PredictRequest(scenario_id=scenario_id, date=on_date, persist=False)
    else:
        payload = PredictRequest(
            target_water_table_depth_cm=wtd_cm if wtd_cm is not None else 20.0,
            date=on_date,
            persist=False,
        )
    resp = prediction_service.run(db, payload)
    return resp
