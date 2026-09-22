# Ajuste de hiperparametros (tuning)

Candidatos: **Random Forest** y **Stacking Ensemble** (los mas estables en la
validacion cruzada) + **XGBoost** (opcional). Busqueda con GridSearchCV /
RandomizedSearchCV y **TimeSeriesSplit(5)** como CV interna
(nunca KFold aleatorio). Tuning solo sobre train 2016-2017; holdout 2018 intacto.

## Espacio de busqueda

### Random Forest - regresion

- `estimator__n_estimators`: [200, 400, 800]
- `estimator__max_depth`: [None, 12, 24]
- `estimator__min_samples_leaf`: [1, 2, 4]
### Random Forest - clasificacion

- `n_estimators`: [200, 400, 800]
- `max_depth`: [None, 12, 24]
- `min_samples_leaf`: [1, 2, 4]
### Stacking - regresion

- `estimator__final_estimator`: [LinearRegression, Ridge(alpha=1.0), Ridge(alpha=10.0)]
- `estimator__rf__n_estimators`: [150, 300]
- `estimator__rf__max_depth`: [None, 12]
- `estimator__xgb__max_depth`: [3, 5]
- `estimator__xgb__learning_rate`: [0.03, 0.1]
- `estimator__svr__regressor__svr__C`: [1.0, 10.0]
### Stacking - clasificacion

- `final_estimator`: [LogisticRegression(C=0.1), LogisticRegression(C=1.0), LogisticRegression(C=10.0)]
- `cv`: [StratifiedKFold(n_splits=5, random_state=None, shuffle=False)]
- `rf__n_estimators`: [150, 300]
- `xgb__max_depth`: [3, 5]
- `xgb__learning_rate`: [0.03, 0.1]
- `svc__svc__C`: [1.0, 10.0]
### XGBoost - regresion

- `estimator__n_estimators`: [300, 500, 800]
- `estimator__max_depth`: [3, 4, 6]
- `estimator__learning_rate`: [0.03, 0.05, 0.1]
- `estimator__subsample`: [0.8, 1.0]
- `estimator__colsample_bytree`: [0.8, 1.0]
### XGBoost - clasificacion

- `n_estimators`: [300, 500, 800]
- `max_depth`: [3, 4, 6]
- `learning_rate`: [0.03, 0.05, 0.1]
- `subsample`: [0.8, 1.0]
- `colsample_bytree`: [0.8, 1.0]

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

### Random Forest (tuned)
```
{
  "regresion": {
    "estimator__max_depth": 12,
    "estimator__min_samples_leaf": 4,
    "estimator__n_estimators": 800
  },
  "clasificacion": {
    "max_depth": null,
    "min_samples_leaf": 1,
    "n_estimators": 400
  }
}
```
### Stacking Ensemble (tuned)
```
{
  "regresion": {
    "estimator__xgb__max_depth": 5,
    "estimator__xgb__learning_rate": 0.03,
    "estimator__svr__regressor__svr__C": 10.0,
    "estimator__rf__n_estimators": 150,
    "estimator__rf__max_depth": null,
    "estimator__final_estimator": "Ridge(alpha=10.0)"
  },
  "clasificacion": {
    "xgb__max_depth": 3,
    "xgb__learning_rate": 0.03,
    "svc__svc__C": 1.0,
    "rf__n_estimators": 300,
    "final_estimator": "LogisticRegression(C=1.0)",
    "cv": "StratifiedKFold(n_splits=5, random_state=None, shuffle=False)"
  }
}
```
### Gradient Boosting (XGBoost) (tuned)
```
{
  "regresion": {
    "estimator__subsample": 1.0,
    "estimator__n_estimators": 800,
    "estimator__max_depth": 3,
    "estimator__learning_rate": 0.1,
    "estimator__colsample_bytree": 1.0
  },
  "clasificacion": {
    "subsample": 0.8,
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8
  }
}
```

## Comparativa base vs tuneado (holdout 2018)

`reg_score_norm` y `combined_score` se **recalculan sobre el pool ampliado**
(5 modelos base + tuneados), por lo que los valores absolutos difieren de los del
Paso 5 (que tenia 5 modelos). La comparacion base-vs-tuneado de esta tabla si es
homogenea (mismo pool).

| model | nee_r2 | fch4_r2 | f1_macro | auc | recall_sumidero | reg_score_norm | cls_score | combined_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.2062 | 0.8309 | 0.7891 | 0.9113 | 0.65 | 0.9034 | 0.8502 | 0.8768 |
| Gradient Boosting (XGBoost) | 0.157 | 0.8438 | 0.803 | 0.9023 | 0.75 | 0.8164 | 0.8526 | 0.8345 |
| SVR / SVM | 0.219 | 0.7197 | 0.7483 | 0.9141 | 0.5 | 0.6676 | 0.8312 | 0.7494 |
| Stacking Ensemble | 0.2155 | 0.8368 | 0.827 | 0.923 | 0.8083 | 0.9398 | 0.875 | 0.9074 |
| CNN-LSTM | 0.1991 | 0.4609 | 0.7989 | 0.855 | 0.95 | 0.0 | 0.827 | 0.4135 |
| Random Forest (tuned) | 0.2303 | 0.831 | 0.7891 | 0.9113 | 0.65 | 0.9613 | 0.8502 | 0.9057 |
| Stacking Ensemble (tuned) | 0.2355 | 0.8419 | 0.8175 | 0.9208 | 0.825 | 1.0 | 0.8691 | 0.9346 |
| Gradient Boosting (XGBoost) (tuned) | 0.1099 | 0.8479 | 0.8175 | 0.9127 | 0.7917 | 0.7135 | 0.8651 | 0.7893 |

## Decision

**Stacking Ensemble (tuned)** mejora el campeon base (0.9346 > 0.9074). Se actualiza `modelo_final.pkl` y el registro en `ml_models` (version 1.1-tuned, is_active=true).

*(generado por `python -m ml.training.tuning`)*
