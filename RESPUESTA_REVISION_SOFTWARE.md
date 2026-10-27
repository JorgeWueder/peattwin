# Respuesta a la revisión por pares: observaciones de software

Cada punto indica si la observación se **rebate** (ya está resuelto en el artículo o en el código), se **acepta** (hay que corregir el texto) o se **atiende con un experimento nuevo**. Los valores salen del código del repositorio y de los CSV de `ml/training/`.

## 1. Observaciones rebatibles: el dato existe y solo falta citarlo

| ID | Observación | Evidencia | Texto propuesto para el artículo |
|---|---|---|---|
| Código | "No se proporciona código ni repositorio" | Repositorio público `github.com/JorgeWueder/peattwin` con `requirements.txt` fijado | Añadir en "Disponibilidad de datos y código" la URL y, si es posible, un DOI de Zenodo. |
| Semillas | "No se especifican" | `SEED = 42` ([config.py:48](ml/training/config.py)); todos los modelos usan `random_state=C.SEED`; la CNN-LSTM usa `keras.utils.set_random_seed(42)` | "Todos los modelos y la búsqueda aleatoria usan la semilla 42." |
| FON-06 | Iteraciones de la búsqueda aleatoria | [tuning.py:170-190](ml/training/tuning.py): Stacking `n_iter = 8`, XGBoost `n_iter = 15`; Random Forest usa rejilla completa (27 combinaciones). Validación interna `TimeSeriesSplit(5)` | Añadir una fila a la Tabla 6 con n_iter y semilla. |
| FON-07 | Parada temprana de la CNN-LSTM | [run_training.py:145](ml/training/run_training.py): `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)`. Validación = último 15 % del tramo de entrenamiento, sin barajar | "Parada temprana sobre la pérdida de validación (último 15 % del entrenamiento), paciencia 15 y restauración de los mejores pesos." En la validación cruzada la paciencia es 10 ([cross_validation.py:148](ml/training/cross_validation.py)); conviene decirlo. |
| FON-08 | Pesos de clase de la CNN-LSTM | [run_training.py:123-131](ml/training/run_training.py): `w_k = n / (2·n_k)` | "Peso por muestra w_k = n/(2·n_k), calculado en el tramo de ajuste (≈1,5 para sumidero y ≈0,75 para fuente)." Verificar los valores exactos antes de escribirlos. |
| FON-09 | Parámetros de estandarización | `StandardScaler` ajustado solo con entrenamiento ([run_training.py:112-114](ml/training/run_training.py)) y serializado en `modelo_cnn_lstm_scalers.pkl` | "Los escaladores se ajustan en entrenamiento y se serializan con el modelo." |
| FON-10 | "Versiones que no existen" | Coinciden con [requirements.txt](requirements.txt) y con las usadas al serializar `modelo_final.pkl` | Añadir la fecha de instalación y el hash del commit. No se cambia nada: son las versiones realmente usadas. |
| FON-15 | Retardo 7 de Diebold-Mariano | [statistical_tests.py:208](ml/training/statistical_tests.py): `lag = max(h-1, round(n^(1/3)))`. Con n = 365 da 7 | "El retardo sigue la regla de truncamiento ⌊n^(1/3)⌉ = 7 (n = 365)." No hace falta un análisis aparte. |
| FON-14 | Conteo 13 vs 16 en la Tabla 3 | La tabla lista 13 controladores (WTD, TS_1, TS_2 y 10 meteorológicos), 6 retardos, 8 móviles y 6 estacionales: 13+6+8+6 = 33. El paso de entrenamiento sustituye TS_1–TS_5 por TS_promedio y llega a 32 | El conteo es correcto. Se puede aclarar la nota al pie: "TS_1–TS_5 solo intervienen en el promedio; las dos sondas listadas son las que genera la extracción". Verificar contra `config.py` antes de afirmar esto. |
| Delwiche 2021 | "Citada en el texto, ausente de la lista" | Está completa en "Disponibilidad de datos" (línea previa a REFERENCIAS), pero **no** dentro de la lista formal | Observación parcialmente válida: moverla a REFERENCIAS y citarla en el texto como (Delwiche et al., 2021). |
| Duplicados de referencias | Palviainen, Premrov, Qiu duplicados | En la lista del manuscrito actual aparece cada una una sola vez | Rebatible. Probable lectura de una versión anterior. |
| Erratas | "Zamekow", "PeaTwin", "tuberas", "ameriFlux" | No aparecen en `G1_DeLaCruz_Ortiz_Moreno_Aguilar.docx` | Rebatible. Ya corregidas. |

## 2. Observaciones atendidas con un experimento

### FON-02 · Criterio combinado dependiente del conjunto (`ml/training/tuning_resultados.csv`)

Se recalculó el ranking de los ocho modelos sin normalizar: `0,5·R² medio + 0,5·(0,5·F1 + 0,5·AUC)`, y con solo la componente de clasificación.

| Criterio | Primero | Segundo |
|---|---|---|
| Original (min-max) | Stacking Ensemble | Stacking (ajustado) |
| Sin normalizar | Stacking (ajustado) | Stacking Ensemble |
| Solo clasificación | Stacking Ensemble | Stacking (ajustado) |

El ganador es la familia Stacking en los tres casos. Con el criterio sin normalizar el primer puesto pasa de la versión base a la ajustada (0,7039 frente a 0,7006), una diferencia pequeña. Proponer añadir esta tabla como análisis de sensibilidad. Hay que reconocer que la normalización min-max hace que el valor absoluto del criterio dependa del conjunto, pero la selección del Stacking no.

### FON-01 · Fuga temporal en las metacaracterísticas del Stacking

Script nuevo: [sensibilidad_stacking.py](ml/training/sensibilidad_stacking.py), resultados en `ml/training/sensibilidad_stacking.csv`. Compara el `KFold(5)` en bloques actual con una versión purgada que deja fuera `gap` días a cada lado del bloque de test. Los bloques de test siguen siendo una partición, así que `cross_val_predict` lo acepta. Holdout 2018:

| gap (días) | R² NEE | R² FCH₄ | F1 macro | AUC |
|---|---|---|---|---|
| 0 (actual) | 0,2155 | 0,8368 | 0,8270 | 0,9230 |
| 7 | 0,2100 | 0,8314 | 0,8283 | 0,9213 |
| 30 | 0,2520 | 0,8386 | 0,8236 | 0,9198 |

La fila gap = 0 reproduce exactamente la Tabla del artículo. Purgar 7 o 30 días no degrada las métricas de forma apreciable, así que no hay indicio de inflado por fuga. Matiz honesto: la purga elimina la contaminación por autocorrelación local, pero el entrenamiento de un bloque temprano sigue usando bloques posteriores. Es una limitación inherente a `cross_val_predict` y hay que decirlo. El holdout 2018 queda intacto en todos los casos.

## 3. Observaciones válidas sin resolver

- **FON-03:** los contrastes se aplican a los modelos base. El artículo ya lo declara; faltaría una segunda batería sobre los modelos ajustados, o dejarlo como limitación explícita.
- **Intervalos de confianza:** no hay. Se puede añadir un bootstrap por bloques sobre el holdout.
- **RMSE del resumen:** el resumen cita 1,261 y 43,80, que no aparecen en una tabla del cuerpo. Hay que añadir una tabla con las métricas del modelo desplegado.
- **"Gemelo digital":** no es de software, pero conviene incorporar la taxonomía de Kritzinger et al. (2018), que hoy no se cita.
