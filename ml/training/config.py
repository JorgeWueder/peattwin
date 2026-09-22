"""Parametros del entrenamiento y la comparacion de modelos."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Entrada: dataset procesado ORIGINAL (no el winsorizado) ---
# Los extremos de NEE/FCH4 son senal geofisica real (pulsos de emision, olas de
# calor) que el gemelo digital debe reproducir; usarlos recortados sesgaria los
# balances de carbono. Por eso se entrena sobre el parquet original.
ML_PARQUET = ROOT / "data" / "processed" / "DE-Zrk_ml_dataset_2016-2018.parquet"
OBS_PARQUET = ROOT / "data" / "processed" / "DE-Zrk_flux_daily_2016-2018.parquet"

# --- Salidas ---
TRAIN_DIR = Path(__file__).resolve().parent
FIG_DIR = TRAIN_DIR / "figuras"
RESULTS_CSV = TRAIN_DIR / "resultados_comparativa.csv"
METRICS_LONG_CSV = TRAIN_DIR / "resultados_detalle_por_objetivo.csv"
RESUMEN_MD = TRAIN_DIR / "RESUMEN_entrenamiento.md"

# --- Validacion cruzada temporal (cross_validation.py) ---
CV_N_SPLITS = 5          # nº de folds por defecto (parametro --folds)
CV_GAP = 0               # dias descartados entre train y test de cada fold (--gap)
CV_PER_FOLD_CSV = TRAIN_DIR / "cv_por_fold.csv"
CV_RESUMEN_CSV = TRAIN_DIR / "cv_resumen.csv"

# --- Ajuste de hiperparametros (tuning.py) ---
TUNING_TSCV_SPLITS = 5   # TimeSeriesSplit interno del tuning
TUNING_CSV = TRAIN_DIR / "tuning_resultados.csv"
TUNING_MD = TRAIN_DIR / "TUNING_resumen.md"

# --- Pruebas estadisticas (statistical_tests.py) ---
STATS_CSV = TRAIN_DIR / "statistical_tests_resultados.csv"
STATS_MD = TRAIN_DIR / "PRUEBAS_ESTADISTICAS_resumen.md"
STATS_FIG = FIG_DIR / "statistical_tests_distribuciones.png"
STATS_ALPHA = 0.05

# Drivers de radiacion / energia que deben estar presentes como features
# (principal control de la GPP / fotosintesis y por tanto de NEE).
RADIATION_FEATURES = ["SW_IN_F", "VPD_F", "NETRAD_F"]

MODELS_DIR = ROOT / "ml" / "models"
FINAL_H5 = MODELS_DIR / "modelo_final.h5"
FINAL_PKL = MODELS_DIR / "modelo_final.pkl"
FINAL_META = MODELS_DIR / "modelo_final_metadata.json"

SEED = 42

# --- Objetivos ---
TARGET_NEE = "NEE_F_ANNOPTLM"          # regresion CO2 (gC m-2 d-1)
TARGET_FCH4 = "FCH4_F_ANNOPTLM"        # regresion CH4 (nmol m-2 s-1)
TARGET_CLS = "clase_balance_carbono_cod"  # 1 = sumidero (clase POSITIVA), 0 = fuente
CLASS_NAMES = {0: "fuente", 1: "sumidero"}

# --- Colinealidad TS_1..TS_5 (hallazgo del EDA: r > 0.95) ---
# Se sustituyen por una unica feature TS_promedio = media(TS_1..TS_5). Es clave
# para SVR/SVM: features casi identicas inflan/!desestabilizan los coeficientes y
# el termino de regularizacion. Se recalculan tambien sus lags/medias moviles a
# partir de la serie diaria continua de TS_promedio.
SOIL_TEMP_RAW = ["TS_1", "TS_2", "TS_3", "TS_4", "TS_5"]
TS_MEAN = "TS_promedio"
DROP_FEATURES = [
    "TS_1", "TS_2",
    "TS_1_lag1", "TS_1_lag3", "TS_1_lag7",
    "TS_1_roll7_mean", "TS_1_roll30_mean",
]

# --- CNN-LSTM ---
WINDOW = 30              # dias de historia por ventana (patrones cortos + dependencia larga)
CNN_LSTM_EPOCHS = 120
CNN_LSTM_BATCH = 32
CNN_LSTM_VAL_FRAC = 0.15   # ultimo 15 % del tramo de train como validacion (temporal)
CNN_LSTM_CLS_LOSS_WEIGHT = 2.0  # se prima la cabeza de clasificacion (clases desbalanceadas)

# --- Criterio de "mejor modelo combinado" ---
# combined = W_REG * reg_score_norm + W_CLS * cls_score
#   reg_score_norm : R2 medio (NEE, FCH4) normalizado min-max entre los 5 modelos
#   cls_score      : 0.5 * F1_macro + 0.5 * AUC   (NO accuracy: con 32/68 de
#                    desbalance, "siempre fuente" da acc ~0.68 sin aprender nada;
#                    F1_macro y AUC penalizan ese sesgo)
COMBINED_W_REG = 0.5
COMBINED_W_CLS = 0.5

MODEL_REGISTRY = {
    "Random Forest": {"framework": "scikit-learn", "algorithm_type": "CLASSICAL"},
    "Gradient Boosting (XGBoost)": {"framework": "xgboost", "algorithm_type": "CLASSICAL"},
    "SVR / SVM": {"framework": "scikit-learn", "algorithm_type": "CLASSICAL"},
    "CNN-LSTM": {"framework": "keras-tensorflow", "algorithm_type": "HYBRID"},
    "Stacking Ensemble": {"framework": "scikit-learn", "algorithm_type": "HYBRID"},
}
