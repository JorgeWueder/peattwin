"""Constructores de los modelos clasicos y del stacking.

Escalado:
  * RF y XGBoost son invariantes a la escala -> sin StandardScaler.
  * SVR/SVM SI necesitan escalado -> Pipeline(StandardScaler, SVx). Ademas para
    SVR se escala tambien el objetivo (TransformedTargetRegressor) porque NEE y
    FCH4 tienen escalas muy distintas.
Desbalance de clases (233 sumidero / 469 fuente en train):
  * RF/SVC: class_weight balanceado.
  * XGBoost: scale_pos_weight = n_fuente / n_sumidero.
"""
from __future__ import annotations

import warnings

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from ml.training import config as C

# CV interno del stacking para generar meta-features out-of-fold.
# KFold en bloques SIN barajar (shuffle=False): cada muestra cae en exactamente
# un fold de test (particion, requisito de cross_val_predict). TimeSeriesSplit NO
# sirve aqui porque deja el primer bloque sin test. La evaluacion FINAL sigue
# siendo el holdout temporal 2018.
_CV = KFold(n_splits=5, shuffle=False)


# --------------------------------------------------------------------- SVR / SVM
def svr_regressor() -> TransformedTargetRegressor:
    base = Pipeline([("scaler", StandardScaler()),
                     ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale"))])
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


def svc_classifier() -> Pipeline:
    # SVC(probability=True) usa escalado de Platt (CV interno) para dar
    # predict_proba, necesario para ROC/AUC. En sklearn 1.9 el parametro esta
    # marcado como deprecado (se elimina en 1.11); la ruta de migracion es
    # CalibratedClassifierCV(SVC(), ensemble=False). Se mantiene probability=True
    # porque da un punto de operacion (umbral 0.5) mas comparable con el resto de
    # modelos; se silencia solo ese FutureWarning.
    warnings.filterwarnings(
        "ignore", message=".*probability.*parameter was deprecated.*",
        category=FutureWarning,
    )
    return Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(C=10.0, kernel="rbf", gamma="scale",
                                 class_weight="balanced", probability=True,
                                 random_state=C.SEED))])


# --------------------------------------------------------------- Random Forest
def rf_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1,
                                 random_state=C.SEED)


def rf_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1,
                                  random_state=C.SEED, class_weight="balanced_subsample")


# -------------------------------------------------------------------- XGBoost
def xgb_regressor() -> XGBRegressor:
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        random_state=C.SEED, n_jobs=-1, objective="reg:squarederror")


def xgb_classifier(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         random_state=C.SEED, n_jobs=-1, eval_metric="logloss",
                         scale_pos_weight=scale_pos_weight)


# ----------------------------------------------------- Stacking (modelo hibrido)
def stacking_regressor() -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=C.SEED)),
            ("xgb", xgb_regressor()),
            ("svr", svr_regressor()),
        ],
        final_estimator=LinearRegression(),   # meta-modelo lineal que pondera los 3
        cv=_CV, n_jobs=1, passthrough=False,
    )


def stacking_classifier(scale_pos_weight: float) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                          random_state=C.SEED, class_weight="balanced_subsample")),
            ("xgb", xgb_classifier(scale_pos_weight)),
            ("svc", svc_classifier()),
        ],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
        stack_method="predict_proba", cv=_CV, n_jobs=1, passthrough=False,
    )
