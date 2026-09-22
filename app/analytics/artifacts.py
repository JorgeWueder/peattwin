"""Lectura de los artefactos que ya produjo el pipeline ML.

Ningun cargador reejecuta nada: todo sale de los CSV, JSON, PKL y Markdown que
`ml/` dejo en disco. Todos toleran que el fichero falte —devuelven un DataFrame
vacio, `None` o lista vacia— porque la seccion Motor IA debe explicar que falta y
como regenerarlo, no reventar con un traceback.

Las rutas se toman de `app.reports.common`, que ya las tenia definidas para los
informes: no se duplican aqui.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.reports import common as C
# Rutas canonicas de los artefactos por algoritmo: se importan del propio
# pipeline para que la app y el entrenamiento no puedan divergir. Import a nivel
# de modulo (nunca dentro de una funcion) para no sombrear nombres globales.
from ml.training.run_training import _bundle_paths

# --- artefactos, agrupados por etapa del pipeline ---------------------------
EDA_TBL = C.EDA_TBL
EDA_FIG = C.EDA_FIG
TRAIN_DIR = C.TRAIN_DIR
TRAIN_FIG = C.TRAIN_FIG
MODELS_DIR = C.settings.models_dir
LOGS_DIR = C.settings.project_root / "ml" / "logs"

_ML_PARQUET = C.PROCESSED / "DE-Zrk_ml_dataset_2016-2018.parquet"

# Nombres de los 5 algoritmos, tal y como los escribe el pipeline.
ALGORITMOS = [
    "Random Forest",
    "Gradient Boosting (XGBoost)",
    "SVR / SVM",
    "CNN-LSTM",
    "Stacking Ensemble",
]


def _csv(path: Path, **kw) -> pd.DataFrame:
    """Lee un CSV y devuelve un DataFrame vacio si no existe o esta corrupto."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


# ------------------------------------------------------------------------ EDA
def descriptivos() -> pd.DataFrame:
    return _csv(EDA_TBL / "01_descriptivos.csv")


def outliers() -> pd.DataFrame:
    return _csv(EDA_TBL / "02_outliers_iqr.csv")


def correlacion(metodo: str = "pearson") -> pd.DataFrame:
    """Matriz cuadrada de correlacion (15x15) con las variables como indice."""
    return _csv(EDA_TBL / f"03_correlacion_{metodo}.csv", index_col=0)


def normalidad() -> pd.DataFrame:
    return _csv(EDA_TBL / "06_normalidad.csv")


def clase_por_mes() -> pd.DataFrame:
    return _csv(EDA_TBL / "05_clase_balance_por_mes.csv")


# -------------------------------------------------------- entrenamiento y CV
def comparativa() -> pd.DataFrame:
    return _csv(TRAIN_DIR / "resultados_comparativa.csv")


def cv_resumen() -> pd.DataFrame:
    return _csv(TRAIN_DIR / "cv_resumen.csv")


def cv_por_fold() -> pd.DataFrame:
    return _csv(TRAIN_DIR / "cv_por_fold.csv")


def tuning() -> pd.DataFrame:
    return _csv(TRAIN_DIR / "tuning_resultados.csv")


def pruebas_estadisticas() -> pd.DataFrame:
    return _csv(TRAIN_DIR / "statistical_tests_resultados.csv")


def secciones_estadisticas() -> list[str]:
    """Secciones REALES del CSV, leidas del fichero (no codificadas a mano)."""
    df = pruebas_estadisticas()
    if df.empty or "seccion" not in df.columns:
        return []
    return sorted(df["seccion"].dropna().unique().tolist())


def folds_disponibles() -> list[int]:
    """Folds REALES que se corrieron, leidos del CSV."""
    df = cv_por_fold()
    if df.empty or "fold" not in df.columns:
        return []
    return sorted(int(f) for f in df["fold"].dropna().unique())


# ------------------------------------------------------------ modelo ganador
def metadata_modelo() -> dict | None:
    """`ml/models/modelo_final_metadata.json` (el JSON trae NaN, que json acepta)."""
    path = MODELS_DIR / "modelo_final_metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:  # pickle corrupto o incompatible
        return None


def pesos_meta(path: Path | None = None) -> pd.DataFrame:
    """Coeficientes del meta-modelo del Stacking sobre sus 3 estimadores base.

    Es lo mas cercano a una "importancia" del ensemble: cuanto pesa cada modelo
    base en la combinacion final. Vacio si el modelo no es un Stacking.
    """
    bundle = _bundle(path or (MODELS_DIR / "modelo_final.pkl"))
    if bundle is None:
        return pd.DataFrame()
    reg = bundle.get("regressor_nee")
    final = getattr(reg, "final_estimator_", None)
    coefs = getattr(final, "coef_", None)
    nombres = [n for n, _ in getattr(reg, "estimators", [])]
    if coefs is None or not nombres:
        return pd.DataFrame()
    valores = list(pd.Series(coefs).astype(float))[: len(nombres)]
    return pd.DataFrame.from_records(
        [{"modelo_base": n, "peso": v} for n, v in zip(nombres, valores)]
    )


def importancias_rf(path: Path | None = None, top: int = 15) -> pd.DataFrame:
    """`feature_importances_` del Random Forest base dentro del Stacking.

    El ensemble como tal no expone importancias globales; se usa la del RF base
    como proxy y la interfaz lo dice explicitamente.
    """
    bundle = _bundle(path or (MODELS_DIR / "modelo_final.pkl"))
    if bundle is None:
        return pd.DataFrame()
    reg = bundle.get("regressor_nee")
    nombres_feat = list(getattr(reg, "feature_names_in_", []))
    estimadores = dict(getattr(reg, "named_estimators_", {}) or {})
    rf = estimadores.get("rf", reg)
    imp = getattr(rf, "feature_importances_", None)
    if imp is None or not nombres_feat:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(
        [{"variable": n, "importancia": float(v)} for n, v in zip(nombres_feat, imp)]
    )
    return df.sort_values("importancia", ascending=False).head(top).reset_index(drop=True)


# ------------------------------------------------- modelos guardados en disco
@dataclass(frozen=True)
class ModeloEnDisco:
    nombre: str
    ruta: Path
    formato: str          # "pkl" | "h5"
    es_desplegado: bool
    modificado: datetime


def modelos_en_disco() -> list[ModeloEnDisco]:
    """Artefactos disponibles para predecir: el desplegado y las copias de `save_all`."""
    salida: list[ModeloEnDisco] = []
    final = MODELS_DIR / "modelo_final.pkl"
    final_h5 = MODELS_DIR / "modelo_final.h5"
    meta = metadata_modelo() or {}
    desplegado = str(meta.get("winner", ""))

    for path, fmt in ((final, "pkl"), (final_h5, "h5")):
        if path.exists():
            salida.append(
                ModeloEnDisco(
                    nombre=f"{desplegado or 'Modelo final'} — desplegado",
                    ruta=path,
                    formato=fmt,
                    es_desplegado=True,
                    modificado=_mtime(path),
                )
            )
            break

    for nombre in ALGORITMOS:
        paths = _bundle_paths(nombre)
        for clave, fmt in (("pkl", "pkl"), ("h5", "h5")):
            if paths[clave].exists():
                salida.append(
                    ModeloEnDisco(
                        nombre=nombre,
                        ruta=paths[clave],
                        formato=fmt,
                        es_desplegado=False,
                        modificado=_mtime(paths[clave]),
                    )
                )
                break
    return salida


# --------------------------------------------------------------- narrativa md
_RESUMENES = {
    "eda": C.EDA_DIR / "EDA_DE-Zrk_resumen.md",
    "entrenamiento": TRAIN_DIR / "RESUMEN_entrenamiento.md",
    "tuning": TRAIN_DIR / "TUNING_resumen.md",
    "estadisticas": TRAIN_DIR / "PRUEBAS_ESTADISTICAS_resumen.md",
}


def resumen_md(nombre: str) -> str | None:
    path = _RESUMENES.get(nombre)
    return C.read_text(path) if path is not None else None


# ------------------------------------------------------------- trazabilidad
def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()


#  etapa · artefacto principal · comando que lo regenera
_ETAPAS: list[tuple[str, Path, str]] = [
    ("ETL", _ML_PARQUET, "python -m ml.etl.run_etl"),
    ("Carga en BD", C.PROCESSED / "DE-Zrk_etl_metadata.json", "python -m ml.etl.load_to_db"),
    ("EDA", EDA_TBL / "01_descriptivos.csv", "python -m ml.eda.run_eda"),
    ("Entrenamiento", TRAIN_DIR / "resultados_comparativa.csv",
     "python -m ml.training.run_training --save-all"),
    ("Modelo desplegado", MODELS_DIR / "modelo_final.pkl",
     "python -m ml.training.run_training --save-all"),
    ("Validacion cruzada", TRAIN_DIR / "cv_por_fold.csv",
     "python -m ml.training.cross_validation --folds 5"),
    ("Tuning", TRAIN_DIR / "tuning_resultados.csv", "python -m ml.training.tuning"),
    ("Pruebas estadisticas", TRAIN_DIR / "statistical_tests_resultados.csv",
     "python -m ml.training.statistical_tests"),
]


def procedencia() -> pd.DataFrame:
    """Estado de frescura de cada etapa: existe, cuando y si quedo desfasada.

    "Desfasado" = el artefacto es ANTERIOR al parquet del dataset, es decir, se
    calculo con datos mas viejos que los actuales.
    """
    ref = _mtime(_ML_PARQUET) if _ML_PARQUET.exists() else None
    filas = []
    for etapa, path, comando in _ETAPAS:
        existe = path.exists()
        cuando = _mtime(path) if existe else None
        desfasado = bool(existe and ref is not None and cuando < ref and path != _ML_PARQUET)
        filas.append(
            {
                "Etapa": etapa,
                "Artefacto": path.name,
                "Estado": "desfasado" if desfasado else ("ok" if existe else "falta"),
                "Generado": cuando.strftime("%Y-%m-%d %H:%M") if cuando else "—",
                "Tamano": f"{path.stat().st_size / 1024:,.0f} KB" if existe else "—",
                "Comando": comando,
            }
        )
    # from_records sobre lista de dicts: inmune a longitudes desiguales.
    return pd.DataFrame.from_records(filas)


def logs_disponibles() -> list[Path]:
    """Ficheros de `ml/logs/`, del mas reciente al mas antiguo."""
    if not LOGS_DIR.exists():
        return []
    return sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)


def leer_log(path: Path, max_lineas: int = 2000) -> str:
    try:
        lineas = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"No se pudo leer {path.name}: {exc}"
    if len(lineas) > max_lineas:
        omitidas = len(lineas) - max_lineas
        lineas = [f"[... {omitidas} lineas anteriores omitidas ...]"] + lineas[-max_lineas:]
    return "\n".join(lineas)
