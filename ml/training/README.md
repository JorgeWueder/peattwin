# Entrenamiento y comparación de modelos — DE-Zrk

Predice los flujos de carbono diarios (CO₂ = `NEE_F_ANNOPTLM`, CH₄ =
`FCH4_F_ANNOPTLM`) y la clase de balance (`clase_balance_carbono`: sumidero /
fuente) a partir del **dataset procesado ORIGINAL** (no el winsorizado — los
extremos de flujo son señal geofísica real que el gemelo digital debe capturar).

```bash
python -m ml.training.run_training            # entrena, compara, guarda, registra en BD
python -m ml.training.run_training --no-db     # sin registro en PostgreSQL
```

## Preparación de features

- **`TS_1..TS_5` → `TS_promedio`** (media del perfil). El EDA encontró
  colinealidad r > 0.95 entre las 5 sondas; features casi idénticas
  desestabilizan los coeficientes de SVR/SVM y su término de regularización.
  Se recalculan también `TS_promedio_lag{1,3,7}` y `TS_promedio_roll{7,30}_mean`
  sobre la serie diaria continua.
- 32 features finales: meteo gap-filled (`_F`), `WTD` + lags/medias móviles,
  `TS_promedio` + derivadas, estacionales (`doy/month` + seno/coseno).
- Se excluyen todas las columnas de flujo como predictores (fuga de información).

## Evaluación

Holdout **temporal** (nunca aleatorio): train = 2016–2017 (702 días),
test = 2018 (365 días). El split ya viene en la columna `split` del parquet.

| Modelo | Familia | Regresión | Clasificación |
|---|---|---|---|
| 1. Random Forest | clásico | `RandomForestRegressor` ×2 | `RandomForestClassifier` (`class_weight` balanceado) |
| 2. Gradient Boosting | clásico | `XGBRegressor` ×2 | `XGBClassifier` (`scale_pos_weight` = 469/233) |
| 3. SVR / SVM | clásico | `SVR` (RBF) + escalado de X y de y | `SVC` (RBF, `class_weight` balanceado, `probability`) |
| 4. CNN-LSTM | híbrido | cabeza `regresion` (Dense 2) | cabeza `clasificacion` (Dense 1 sigmoide) |
| 5. Stacking Ensemble | híbrido | `StackingRegressor`(RF+XGB+SVR → LinearRegression) | `StackingClassifier`(RF+XGB+SVC → LogisticRegression) |

**CNN-LSTM** (Keras/TensorFlow): ventanas de 30 días →
`Conv1D` causal ×2 → `MaxPooling1D` → `LSTM(64)` → `LSTM(32)` → Dense compartida →
**dos cabezas** (regresión de [NEE, FCH4] + clasificación sumidero/fuente).
Pérdida = MSE + 2·BCE; `sample_weight` por clase para el desbalance.

**Stacking**: el CV interno que genera las meta-features usa `KFold` en bloques
(sin barajar); `TimeSeriesSplit` no sirve ahí porque `cross_val_predict` exige
una partición completa. La evaluación final sigue siendo el holdout 2018.

## Nota sobre el R² de NEE (bajo en los 5 modelos: 0.16–0.22)

Verificado: `SW_IN_F`, `VPD_F` y `NETRAD_F` **ya están** en las features (desde el
ETL). El bajo R² de NEE no se debe a un driver de radiación ausente, sino a que
**NEE = GPP − RECO** (diferencia de dos flujos grandes que casi se cancelan →
baja SNR) y a que el dataset **no tiene variables de vegetación** (NDVI, LAI,
EVI): la radiación explica la envolvente estacional de la GPP pero no la
variación diaria de la capacidad fotosintética del dosel. FCH4, controlado por
temperatura del suelo y nivel freático (ambos medidos), es mucho más predecible.
Desarrollo completo en `RESUMEN_entrenamiento.md`.

## Métricas registradas (por modelo)

- **Tiempo de entrenamiento (s)** = suma de todos los `fit` del modelo
  (regresor NEE + regresor FCH4 + clasificador; el CNN-LSTM es un único `fit`).
- **Regresión** (NEE y FCH4 por separado): RMSE, MAE, R² (+ NSE, KGE para la BD;
  KGE de NEE = `nan` porque su media ≈ 0 hace explotar el término β).
- **Clasificación**: accuracy, balanced_accuracy, precision/recall/F1 **macro y
  por clase**, matriz de confusión, curva ROC, AUC.

> **El desbalance (233 sumidero / 469 fuente, ≈32/68) importa.** Un clasificador
> trivial "siempre fuente" logra accuracy ≈ 0.68 sin aprender nada. Por eso se
> reportan F1 y recall de la clase *sumidero* de forma destacada y el criterio de
> ganador **no** usa accuracy.

## Criterio de "mejor modelo combinado"

```
combined = 0.5 · reg_score_norm  +  0.5 · cls_score
  reg_score_norm = R² medio(NEE, FCH4) normalizado min-max entre los 5 modelos
  cls_score      = 0.5 · F1_macro  +  0.5 · AUC
```

50 % regresión / 50 % clasificación. En regresión se normaliza el R² medio para
que sea comparable en escala con la métrica de clasificación. En clasificación se
usan **F1_macro y AUC** (no accuracy) para no premiar el sesgo hacia la clase
mayoritaria.

## Salidas

| Ruta | Contenido |
|---|---|
| `resultados_comparativa.csv` | **Tabla consolidada**: todas las métricas de los 5 modelos + scores + ganador |
| `RESUMEN_entrenamiento.md` | Resumen interpretado para el informe (verificación de features + por qué el R² de NEE es bajo) |
| `figuras/matrices_confusion.png` | Matriz de confusión de cada modelo (holdout 2018) |
| `figuras/curvas_roc_comparacion.png` | Las 5 curvas ROC en un eje + AUC |
| `figuras/heatmap_comparativo_metricas.png` | Heatmap comparativo (verde = mejor) |
| `figuras/comparativa_regresion_clasificacion.png` | Barras R² + líneas de métricas de clasificación |
| `../models/modelo_final.pkl` **o** `.h5` | Modelo ganador (ver abajo) |
| `../models/modelo_final_metadata.json` | Ganador, criterio, métricas, features |

### Formato del modelo ganador

- Si gana el **CNN-LSTM** → `model.save('modelo_final.h5')` (HDF5, formato nativo
  de Keras) + `modelo_final_scalers.pkl`.
- Si gana un **modelo clásico / stacking** → `joblib.dump(..., 'modelo_final.pkl')`.
  **`.h5` NO aplica**: es un formato específico de Keras/TensorFlow (serializa el
  grafo de capas y los pesos de los tensores). Un modelo scikit-learn / XGBoost /
  stacking es un objeto Python normal que se serializa con pickle; `joblib` es el
  estándar de facto para scikit-learn. (Comentario equivalente en
  `run_training.py::save_winner`.)

## Registro en PostgreSQL

Cada uno de los 5 modelos genera 2 filas en `ml_models` (`… - regresión` con
`task=REGRESSION`, `… - clasificación` con `task=CLASSIFICATION`) y 2 filas en
`ml_model_metrics` (`fold = -1` = holdout). El **ganador** queda con
`is_active = true` en sus dos filas (una por tarea → respeta el índice único
parcial "un modelo activo por tarea"). Idempotente: `ON CONFLICT (name, version)`.
El detalle por objetivo (NEE vs FCH4) vive en el CSV; en la BD la fila de
regresión guarda la media de ambos.

---

# Validación cruzada temporal — `cross_validation.py`

```bash
python -m ml.training.cross_validation                 # 5 folds, con CNN-LSTM, registra en BD
python -m ml.training.cross_validation --folds 8 --gap 30
python -m ml.training.cross_validation --no-cnn --no-db   # rápido, sin escribir
```

- **`TimeSeriesSplit`** de scikit-learn, **nunca** `KFold` aleatorio: barajar el
  orden de una serie temporal es fuga de datos (el modelo "vería" el futuro).
  Cada fold: `train = [0..k]`, `test = (k..k+m]`, contiguos y en orden; los folds
  sucesivos amplían el train y desplazan el test hacia adelante.
- **`--folds`** configurable (por defecto **5**). **`--gap`** (por defecto 0)
  descarta N días entre train y test de cada fold para cortar la autocorrelación
  residual de los lags / medias móviles.
- Por **cada modelo y cada fold**: regresión (NEE y FCH4) → RMSE, MAE, R², NSE,
  KGE; clasificación → accuracy, precision/recall/F1 macro, AUC; tiempo de
  entrenamiento. `NSE ≡ R²` por definición (Nash-Sutcliffe con la media
  observada); `KGE(NEE) = nan` porque la media de NEE ≈ 0.

## Salidas

| Ruta | Contenido |
|---|---|
| `cv_por_fold.csv` | Una fila por `(modelo, fold)` con todas las métricas |
| `cv_resumen.csv` | `<métrica>_mean` y `<métrica>_std` por modelo (media ± desv. estándar) |
| `figuras/cv_boxplots_estabilidad.png` | Boxplots de RMSE (NEE y FCH4), F1_macro y R²(FCH4) entre los 5 modelos a través de los folds; **caja más corta = modelo más estable** |
| `ml_model_metrics` | Filas `fold = 0 … n_splits-1` (upsert por `(model_id, fold)`); conviven con `fold = -1` del holdout de `run_training.py` |

## Nota de interpretación (para el informe)

Esta CV mide **estabilidad** sobre toda la serie 2016-2018; es complementaria al
holdout único 2018 de `run_training.py` (guardado como `fold = -1`).

El **fold 0** entrena con solo ~182 días (< medio ciclo estacional) y da R²
negativos en casi todos los modelos: es un hallazgo esperado sobre suficiencia de
datos, no un fallo. Con `--folds 3` cada fold tiene más historia y los resultados
se estabilizan. En la desviación estándar del resumen se ve que **SVR/SVM y
CNN-LSTM son los menos estables** en regresión de FCH4 (std de R² ≈ 0.55 y 0.78),
mientras que **Random Forest y Stacking** son los más consistentes (std ≈ 0.31).

---

# Ajuste de hiperparámetros — `tuning.py`

```bash
python -m ml.training.tuning                       # RF + Stacking + XGBoost, actualiza BD si mejora
python -m ml.training.tuning --models rf,stacking  # sin XGBoost
python -m ml.training.tuning --fast --no-db        # prueba rápida
```

Ajusta **Random Forest** y **Stacking Ensemble** (los más estables en la CV) y,
opcionalmente, **XGBoost**.

- **CV interna del tuning**: `TimeSeriesSplit(5)` de scikit-learn (nunca KFold
  aleatorio). Solo sobre train 2016-2017; el holdout 2018 queda intacto para la
  comparación final.
- **`GridSearchCV`** para Random Forest (espacio pequeño), **`RandomizedSearchCV`**
  para Stacking y XGBoost (espacios grandes).
- Regresores ajustados con `MultiOutputRegressor` sobre `[NEE, FCH4]`,
  `scoring="r2"` (= `reg_score`). Clasificadores con scorer propio
  `0.5·F1_macro + 0.5·AUC` (= `cls_score`; ignora accuracy por el desbalance).

## Espacio de búsqueda

| Modelo | Hiperparámetros |
|---|---|
| **Random Forest** (reg. y clf.) | `n_estimators` ∈ {200, 400, 800} · `max_depth` ∈ {None, 12, 24} · `min_samples_leaf` ∈ {1, 2, 4} |
| **Stacking – meta-modelo** | regresión: `LinearRegression`, `Ridge(α=1)`, `Ridge(α=10)` · clasificación: `LogisticRegression` con `C` ∈ {0.1, 1, 10} (L2 = ridge; `RidgeClassifier` excluido por no tener `predict_proba`) |
| **Stacking – base RF** | `n_estimators` ∈ {150, 250} · `max_depth` ∈ {None, 12} |
| **Stacking – base XGB** | `max_depth` ∈ {3, 5} · `learning_rate` ∈ {0.03, 0.1} |
| **Stacking – base SVR/SVC** | `C` ∈ {1, 10} |
| **XGBoost** (reg. y clf.) | `n_estimators` ∈ {300, 500, 800} · `max_depth` ∈ {3, 4, 6} · `learning_rate` ∈ {0.03, 0.05, 0.1} · `subsample` ∈ {0.8, 1.0} · `colsample_bytree` ∈ {0.8, 1.0} |

Durante el tuning, el CV **interno** del stacking (meta-features) se fija para
acotar el coste del nested-CV: `KFold(3)` en bloques para la regresión y
`StratifiedKFold(3)` sin barajar para la clasificación (garantiza ambas clases en
cada bloque — `sumidero` se concentra en verano). El modelo tuneado conserva esa
configuración.

## Criterio de selección

Idéntico al Paso 5: `combined = 0.5·reg_score_norm + 0.5·cls_score`, con
`reg_score_norm` = R² medio(NEE, FCH4) normalizado min-max sobre el pool completo
(5 modelos base + versiones tuneadas) y `cls_score = 0.5·F1_macro + 0.5·AUC`. Los
valores absolutos difieren de los del Paso 5 porque el pool ahora tiene más
modelos; la comparación base-vs-tuneado sí es homogénea.

## Salidas

| Ruta | Contenido |
|---|---|
| `tuning_resultados.csv` | Pool completo (base + tuneados) con métricas holdout y scores |
| `TUNING_resumen.md` | Espacio de búsqueda, criterio, mejores hiperparámetros, decisión |
| `../models/modelo_final.pkl` | **Se actualiza solo si un modelo tuneado supera el `combined` del campeón base** |
| `ml_models` / `ml_model_metrics` | Si mejora: nueva versión `1.1-tuned` con `is_active = true` (desactiva las `1.0`) |

**Resultado real:** el `Stacking Ensemble (tuned)` con meta-modelo `Ridge(α=10)`
mejora el `combined` del campeón base (0.935 vs 0.907, R²(NEE) 0.216 → 0.236) →
`modelo_final.pkl` y `ml_models` (v `1.1-tuned`) actualizados.

---

# Pruebas estadísticas — `statistical_tests.py`

```bash
python -m ml.training.statistical_tests
python -m ml.training.statistical_tests --from-csv   # usa cv_por_fold.csv en vez de la BD
python -m ml.training.statistical_tests --no-dm      # sin Diebold-Mariano (no reentrena)
```

Compara formalmente los **5 modelos base** (sin tuning) con los resultados de la
CV (`ml_model_metrics`, `fold` 0-4). Métrica de error por `(modelo, fold)`: `rmse`
de las filas de regresión (media RMSE de NEE y FCH4). Comparación secundaria:
`f1` de las filas de clasificación.

| Prueba | Qué hace | Cuándo |
|---|---|---|
| **Shapiro-Wilk** | Normalidad de la distribución de error de cada modelo entre folds + de las diferencias pareadas | Decide paramétrica vs no paramétrica |
| **t pareada** / **Wilcoxon signed-rank** | `Stacking Ensemble` vs cada uno de los otros 4, con corrección de **Holm** | t si normal, Wilcoxon si no |
| **ANOVA** / **Kruskal-Wallis** + **Friedman** | Los 5 modelos a la vez. Friedman = versión de medidas repetidas (correcta para folds pareados) | ANOVA si normal, KW si no |
| **Post-hoc de Nemenyi** | Si Friedman(RMSE) es significativo: matriz de p-values pareados sobre la matriz RMSE fold×modelo (`scikit-posthocs`, con fallback manual vía rango studentizado) + diagrama de diferencias críticas (CD) | ¿qué par(es) explican la diferencia global de Friedman? |
| **Diebold-Mariano** | `Stacking Ensemble` vs `Random Forest` sobre la secuencia de errores diarios del **holdout 2018** (365 días); Newey-West + corrección HLN de muestra pequeña | Significancia de la diferencia en precisión predictiva |

## Salidas

| Ruta | Contenido |
|---|---|
| `statistical_tests_resultados.csv` | Tabla consolidada de p-values (todas las pruebas, incl. matriz de Nemenyi) |
| `PRUEBAS_ESTADISTICAS_resumen.md` | Tablas + **interpretación en texto simple** (para el informe) + nota metodológica |
| `figuras/statistical_tests_distribuciones.png` | Boxplots de RMSE y F1 por fold |
| `figuras/statistical_tests_nemenyi_cd.png` | Diagrama de diferencias críticas (Nemenyi) |
| `figuras/statistical_tests_nemenyi_pmatrix.png` | Heatmap de la matriz de p-values de Nemenyi |

## Resultado real (resumen)

- **Shapiro** rechaza normalidad (una distribución de RMSE y una diferencia
  pareada de F1 con p < 0.05) → se usan pruebas **no paramétricas**.
- **Pareadas (Wilcoxon + Holm)**: **ninguna** comparación alcanza p < 0.05. Con
  5 folds Wilcoxon no puede bajar de p ≈ 0.0625 (límite estructural), ni siquiera
  frente a SVR/CNN-LSTM que son claramente peores.
- **Omnibus**: Kruskal-Wallis p = 0.26 (no significativo), pero **Friedman
  p = 0.004** (sí significativo) — al respetar el bloqueo por fold tiene mucha más
  potencia y confirma que los 5 modelos **no** rinden todos igual.
- **Post-hoc de Nemenyi** (tras Friedman significativo): rangos medios (1 = mejor)
  XGBoost 1.8, RF 2.0, Stacking 2.2, SVR 4.2, CNN-LSTM 4.8; CD = 2.73. Pares
  significativos: **RF vs CNN-LSTM** (p = 0.041) y **XGBoost vs CNN-LSTM**
  (p = 0.023). → La diferencia global de Friedman la produce **CNN-LSTM** (peor
  que los modelos de árboles); entre **Stacking, RF y XGBoost no hay ninguna
  diferencia significativa** (p ≈ 1.0): son equivalentes en RMSE.
- **Diebold-Mariano** `Stacking` vs `Random Forest`: NEE p = 0.75, FCH4 p = 0.14
  → **sin diferencia significativa** en precisión predictiva (aunque el signo
  favorece a Stacking en ambos).
- **Conclusión para el informe**: no se puede afirmar que `Stacking Ensemble` sea
  estadísticamente superior en RMSE a `Random Forest` (diferencias pequeñas,
  pocos folds). Se elige por su **mejor comportamiento conjunto** (F1_macro + AUC),
  **menor varianza entre folds** y el criterio combinado del Paso 5.

## Nota metodológica

Estas pruebas usan los **5 modelos base** de la CV (`version 1.0`, sin tuning). El
modelo **desplegado** (`Stacking Ensemble (tuned)`, `v1.1-tuned`) es un
refinamiento posterior del ganador ya seleccionado; el tuning solo cambió el
meta-modelo (`Ridge(α=10)`), no la arquitectura, así que no se re-somete a esta
batería.
