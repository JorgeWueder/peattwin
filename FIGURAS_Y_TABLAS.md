# Índice de figuras y tablas — PeatTwin

Catálogo de todo lo que el proyecto genera y puede ir al artículo. Cada entrada
lleva **dos bloques separados**, en este orden:

- **Interpretación** — qué significa. El hallazgo, la conclusión, la salvedad.
  Es lo que sostiene el argumento del artículo y va primero.
- **Explicabilidad** — por qué se puede afirmar eso. Qué hay en los ejes, en qué
  unidades, sobre qué datos y cómo se calculó. Es el respaldo auditable.

Es la misma separación y el mismo orden que impone `ui/interpret.py` en la
aplicación y que usan los informes PDF y Word, de modo que los pies de figura
del artículo salen de aquí sin reescribirlos.

Cifras verificadas contra los artefactos en disco el 2026-09-22. Si se reentrena,
hay que volver a leerlas: las de la interfaz se recalculan solas, las de este
documento no.

- **21 figuras** estáticas (13 del EDA + 8 del entrenamiento) y 3 más que los
  informes generan al vuelo.
- **11 tablas** en CSV (6 del EDA + 5 del entrenamiento).

Ambos conteos superan el mínimo de 6 + 6 exigido. La selección de los que
**deberían** ir al artículo está en la sección 1; el inventario completo, en las
secciones 2 y 3.

---

## 1. Selección para el artículo

Seis tablas y seis figuras que, juntas, cuentan el trabajo entero: de dónde
salen los datos, cómo se comportan, qué modelos se comparan, cuál gana, si esa
victoria es estadísticamente real y para qué sirve el resultado.

### Tablas

| ID | Fuente | Qué establece |
|---|---|---|
| **T1** | `ml/eda/tablas/01_descriptivos.csv` | Caracterización del sitio: las dos variables objetivo y sus drivers |
| **T2** | `ml/eda/tablas/06_normalidad.csv` | Justifica el uso de pruebas no paramétricas en todo el estudio |
| **T6** | `ml/training/resultados_comparativa.csv` | Resultado principal: los 5 modelos en el holdout de 2018 |
| **T7** | `ml/training/cv_resumen.csv` | Estabilidad entre folds — separa «ganó» de «ganó por suerte» |
| **T9** | `ml/training/statistical_tests_resultados.csv` | Significación estadística de las diferencias entre modelos |
| **T10** | `ml/training/tuning_resultados.csv` | Efecto del ajuste de hiperparámetros sobre el criterio combinado |

### Figuras

Marcadas con ★ en el inventario de la sección 3.

| ID | Fuente | Qué muestra |
|---|---|---|
| **F1** | `ml/eda/figuras/03_correlacion_pearson_heatmap.png` | Estructura de dependencias entre drivers y objetivos |
| **F2** | `ml/eda/figuras/07_descomposicion_STL_NEE.png` | El balance de carbono lo manda el ciclo anual |
| **F3** | `ml/training/figuras/heatmap_comparativo_metricas.png` | Comparación de los 5 modelos de un vistazo |
| **F4** | `ml/training/figuras/matrices_confusion.png` | Clasificación sumidero/fuente con clase desbalanceada |
| **F5** | `ml/training/figuras/cv_boxplots_estabilidad.png` | Dispersión entre folds: el argumento de estabilidad |
| **F6** | `ml/training/figuras/statistical_tests_nemenyi_cd.png` | Qué diferencias son significativas y cuáles no |

---

## 2. Tablas

### 2.1 EDA — `ml/eda/tablas/` · `python -m ml.eda.run_eda`

#### T1 · `01_descriptivos.csv` — Estadísticos descriptivos

> **Interpretación.** El NEE tiene media casi nula (0.052 gC m⁻² d⁻¹) pero
> mediana positiva (0.284): el sitio es fuente la mayor parte del año y sumidero
> en pulsos intensos de verano que tiran de la media hacia cero. La asimetría
> negativa (−1.21) confirma esa cola de captura. El FCH₄ es el caso opuesto,
> asimetría +1.57 con media 80.4 frente a mediana 22.9 nmol m⁻² s⁻¹: emisión
> baja casi todos los días y pulsos estivales que dominan el total anual. Las
> cinco profundidades de temperatura de suelo tienen prácticamente la misma
> media (10.9–11.0 °C) y desviaciones decrecientes con la profundidad
> (5.46 → 3.56): el suelo amortigua la oscilación superficial, y esa redundancia
> es lo que justifica sustituirlas por `TS_promedio`.

> **Explicabilidad.** 8 variables × 12 estadísticos sobre los 1096 días de la
> serie diaria de DE-Zrk (2016–2018). `n` es el número de días con dato; media y
> mediana van en la unidad de cada variable (NEE en gC m⁻² d⁻¹, FCH₄ en
> nmol m⁻² s⁻¹, TS en °C, WTD en metros sobre la superficie). La asimetría y la
> curtosis de exceso valen 0 en una distribución normal, así que cuanto más se
> alejan de 0, más se aparta la variable de ella.

#### T2 · `06_normalidad.csv` — Contrastes de normalidad

> **Interpretación.** Ninguna de las 10 series pasa los contrastes: p < 0.05 en
> todos los casos y p ≈ 2.5·10⁻²⁶ (NEE) y 2.0·10⁻³⁷ (FCH₄) en los objetivos.
> Los residuos de la descomposición STL tampoco (p ≈ 10⁻⁴³ y 10⁻⁴⁶), lo que
> descarta que la no normalidad sea solo efecto de la estacionalidad. Esta tabla
> es la que obliga a que toda la comparación posterior entre modelos use
> contrastes no paramétricos —Wilcoxon, Friedman, Nemenyi— en lugar de una t de
> Student, y a transformar el FCH₄ (log1p) donde se asuma normalidad.

> **Explicabilidad.** Dos contrastes sobre cada una de las 10 series:
> Shapiro-Wilk (estadístico W, tanto más cercano a 1 cuanto más normal) y
> D'Agostino-Pearson (K², que combina asimetría y curtosis). La hipótesis nula de
> ambos es que los datos proceden de una normal, de modo que un p pequeño la
> **rechaza**. Se incluyen los residuos de la descomposición STL además de las
> variables en bruto, precisamente para poder separar los dos efectos.

#### T3 · `02_outliers_iqr.csv` — Extremos por criterio IQR

> **Interpretación.** Con k = 1.5 el NEE tiene un 8.85 % de valores extremos y
> el FCH₄ un 1.37 %; el WTD, un 4.74 %. Las cinco series de temperatura de suelo
> no tienen ninguno (0.00 %), lo que confirma que son las variables mejor
> comportadas del conjunto. **Los extremos no se eliminan**: son señal
> geofísica real —pulsos de emisión, olas de calor— que el gemelo digital debe
> reproducir, así que el entrenamiento usa el parquet original y no el
> winsorizado. La versión winsorizada se conserva solo como comparación.

> **Explicabilidad.** Se marcan como extremos los valores fuera de
> Q1 − k·IQR y Q3 + k·IQR, con k = 1.5 (criterio habitual) y k = 3.0 (solo
> extremos severos). `pct_1.5` y `pct_3.0` son el porcentaje de días marcados
> con cada criterio, y `n_winsorizados_k3` cuenta cuántos valores recortaría una
> winsorización al límite de k = 3.

#### T4 · `03_correlacion_pearson.csv` y `03_correlacion_spearman.csv`

> **Interpretación.** La radiación (SW_IN_F, NETRAD_F), la temperatura del aire
> y el VPD forman un bloque fuertemente intercorrelacionado, y TS_1…TS_5 son
> casi idénticas entre sí (r > 0.95), lo que justifica promediarlas en
> `TS_promedio`. El FCH₄ va de la mano de la temperatura del suelo; el NEE se
> explica sobre todo por la radiación, con signo negativo (más luz → más
> fotosíntesis → NEE más negativo).

> **Explicabilidad.** Matrices de correlación entre las variables clave. Cada
> celda es el coeficiente r del par fila-columna, de −1 (relación inversa
> perfecta) a +1 (directa perfecta); la diagonal vale 1 por definición y la
> matriz es simétrica. Se dan las dos versiones porque Pearson mide relación
> **lineal** y Spearman opera sobre los rangos, captando cualquier relación
> monótona: con variables tan asimétricas como el FCH₄, la diferencia entre
> ambas es informativa por sí misma.

#### T5 · `05_clase_balance_por_mes.csv` — Clase por mes

> **Interpretación.** El desbalance global (≈ 32 % sumidero) no está repartido:
> la captura se concentra en los meses centrales del año y desaparece en
> invierno. Es la razón de que el holdout sea **temporal y no aleatorio** —un
> corte al azar mezclaría veranos de entrenamiento y test— y de que la accuracy
> no sirva como métrica: un clasificador estacional trivial ya acertaría mucho.

> **Explicabilidad.** 12 filas, una por mes, con la fracción de días clasificados
> como sumidero y como fuente. La clase se deriva del signo del NEE observado:
> sumidero si NEE < 0.

### 2.2 Entrenamiento — `ml/training/` · `python -m ml.training.run_training`

#### T6 · `resultados_comparativa.csv` — Comparación de los 5 modelos

> **Interpretación.** Es el resultado principal. El hallazgo central no es qué
> modelo gana sino el contraste entre objetivos: el CH₄ se predice bien (R² de
> 0.72 a 0.84 en cuatro de los cinco modelos) y el CO₂ mal (R² de 0.16 a 0.22 en
> todos). La explicación es física, no algorítmica: el FCH₄ depende de
> temperatura de suelo y nivel freático, ambos medidos en el sitio, mientras que
> el NEE es la diferencia entre dos flujos grandes (fotosíntesis y respiración) y
> depende del estado de la vegetación, que Zarnekow no mide (sin NDVI ni LAI). Un
> R² ≈ 0.2 para NEE diario es coherente con la literatura de flujos en turberas.
> Gana **Stacking Ensemble** por criterio combinado (mejor F1_macro 0.827 y mejor
> AUC 0.923), pero el mejor R² de NEE es de SVR/SVM (0.219) y el mejor R² de
> FCH₄, de XGBoost (0.844): **ningún modelo domina en todo**, y por eso el
> criterio de selección tiene que ser explícito. CNN-LSTM queda descolgado en
> FCH₄ (R² 0.461) pese a ser el segundo más caro de entrenar.

> **Explicabilidad.** 5 modelos × 34 columnas. Todas las métricas proceden del
> mismo holdout temporal: se entrena con 2016–2017 (702 días) y se evalúa sobre
> 2018 completo (365 días), sin barajar. RMSE y MAE son errores en la unidad del
> objetivo, luego menor es mejor; R², NSE y KGE son adimensionales y valen 1 en
> la predicción perfecta, 0 cuando el modelo iguala a predecir la media y
> negativo si la empeora. Accuracy, F1 y AUC corresponden a la clasificación
> sumidero/fuente. El criterio de selección es
> `combined = 0.5·reg_score_norm + 0.5·(0.5·F1_macro + 0.5·AUC)`.

#### T7 · `cv_resumen.csv` — Validación cruzada, agregado

> **Interpretación.** Mide **estabilidad**, que es distinto de rendimiento. En
> FCH₄, Stacking Ensemble (R² 0.721 ± 0.310) y Random Forest (0.733 ± 0.321) son
> los más consistentes, mientras que CNN-LSTM (0.385 ± 0.778) tiene una
> desviación el doble que su propia media: su resultado depende del fold que le
> toque, lo que lo descarta como modelo desplegable aunque en algún corte puntual
> sea competitivo. La lectura práctica es que un modelo con media algo peor pero
> desviación baja es mejor apuesta que uno con media alta e inestable.

> **Explicabilidad.** 5 modelos × media y desviación típica de cada métrica sobre
> los 5 folds de `TimeSeriesSplit`. Cada fold amplía la ventana de entrenamiento
> y desplaza la de test hacia adelante, de modo que el modelo nunca ve datos
> posteriores a los que predice. La media resume el rendimiento típico; la
> desviación, cuánto varía de un periodo a otro.

#### T8 · `cv_por_fold.csv` — Validación cruzada, detalle

> **Interpretación.** Explica de dónde salen las desviaciones tan altas de T7.
> Los primeros folds entrenan con menos de medio ciclo estacional y producen R²
> negativos en casi todos los modelos —es decir, peores que predecir la media—.
> **No es un fallo de cálculo sino un resultado sobre suficiencia de datos**: por
> debajo de un año de histórico el problema no es aprendible, lo que fija un
> requisito mínimo para replicar el gemelo en otro sitio.

> **Explicabilidad.** Una fila por combinación de modelo y fold, sin promediar.
> Los folds van en orden cronológico creciente, con más histórico de
> entrenamiento cuanto mayor es su número. NSE ≡ R² por definición con esta
> formulación, de ahí que ambas columnas coincidan.

#### T9 · `statistical_tests_resultados.csv` — Pruebas estadísticas

> **Interpretación.** Es la tabla que impide sobrevender el resultado.
> **Solo 3 de los 24 contrastes con veredicto resultan significativos a
> α = 0.05**: Friedman sobre RMSE (p = 0.0037) y dos pares de Nemenyi
> (Random Forest y XGBoost frente a CNN-LSTM). **Ningún par que incluya al
> Stacking es significativo.** Con 5 folds la potencia estadística es muy baja,
> así que la lectura correcta no es «los modelos son equivalentes» sino «con este
> diseño no se puede demostrar que difieran». El orden por RMSE medio de la CV
> (Random Forest 21.54 < XGBoost 22.16 < Stacking 22.35 < SVR 30.69 <
> CNN-LSTM 31.93) **no coincide** con el del criterio combinado, que elige
> Stacking. Que esa diferencia no sea significativa es justamente lo que permite
> decidir por F1/AUC y estabilidad sin contradecir los datos.

> **Explicabilidad.** 43 filas: normalidad (18), pruebas pareadas de Wilcoxon
> (8), ómnibus (4), post-hoc de Nemenyi (11) y Diebold-Mariano (2). Los
> contrastes operan sobre las métricas por fold de la validación cruzada con los
> cinco modelos base y α = 0.05. Shapiro-Wilk decide primero si procede la rama
> paramétrica o la no paramétrica; Wilcoxon compara pares emparejando fold a
> fold, con corrección de Holm por multiplicidad; Friedman pregunta si algún
> modelo difiere del conjunto respetando ese emparejamiento y Kruskal-Wallis hace
> lo propio ignorándolo; Nemenyi identifica después qué pares lo explican.
> Diebold-Mariano es aparte: opera sobre los 365 errores diarios del holdout, con
> varianza de largo plazo de Newey-West y corrección de Harvey-Leybourne-Newbold.
> **Nota de potencia:** con n = 5, el p mínimo alcanzable por un Wilcoxon a dos
> colas es 2/2⁵ = 0.0625, así que esa prueba **no puede** bajar de 0.05 en este
> diseño.

#### T10 · `tuning_resultados.csv` — Ajuste de hiperparámetros

> **Interpretación.** No todo ajuste mejora: en el Stacking el R² de NEE sube de
> 0.216 a 0.236 y el score combinado de 0.907 a 0.935, pero en XGBoost el ajuste
> **empeora** el NEE. Por eso el pipeline solo promueve la versión ajustada si
> supera a la original en el criterio combinado.

> **Explicabilidad.** 8 filas (modelos base y sus versiones *tuned*) × las
> puntuaciones del criterio. Los hiperparámetros se buscan con
> `GridSearchCV`/`RandomizedSearchCV` y validación interna `TimeSeriesSplit(5)`
> aplicada **solo** al tramo de entrenamiento, de modo que el holdout de 2018 no
> interviene en la búsqueda y las cifras de T6 mantienen su validez.

---

## 3. Figuras

### 3.1 EDA — `ml/eda/figuras/` · `python -m ml.eda.run_eda`

| Fichero | Interpretación | Explicabilidad |
|---|---|---|
| `01_histogramas_kde_variables_clave.png` | El NEE se concentra cerca de cero con cola de captura; el FCH₄ es fuertemente asimétrico a la derecha. Ninguna se parece a una normal, lo que anticipa T2. | Histograma de cada variable clave con su estimación de densidad (KDE) superpuesta, en la unidad original de cada una. |
| `02_boxplots_outliers_iqr.png` | Los extremos del NEE y del FCH₄ son pulsos reales, no errores de medida: se conservan para el entrenamiento. | Diagramas de caja con las variables estandarizadas (z-score) para poder verlas en un mismo eje; bigotes a 1.5·IQR. |
| `03_correlacion_pearson_heatmap.png` ★ | Bloque radiación–temperatura–VPD muy intercorrelacionado; TS_1…TS_5 casi idénticas (justifica `TS_promedio`); FCH₄ ligado a temperatura de suelo y NEE a radiación con signo negativo. | Mapa de calor de la matriz de Pearson; color de −1 a +1 con 0 en el centro, valor impreso en cada celda, matriz simétrica y diagonal igual a 1. |
| `04_correlacion_spearman_heatmap.png` | Donde difiere de la anterior, la relación no es lineal pero sí monótona. | La misma matriz calculada sobre rangos en vez de valores, lo que la hace robusta a los extremos. |
| `05_serie_temporal_NEE_CO2.png` | Los tramos de captura se concentran en verano; no hay deriva plurianual visible en tres años. | Serie diaria completa del NEE en gC m⁻² d⁻¹ frente al tiempo, con la línea de cero marcada. |
| `06_serie_temporal_FCH4_CH4.png` | La turbera emite metano todos los días del registro, con pulsos estivales marcados. | Serie diaria completa del FCH₄ en nmol m⁻² s⁻¹; sin línea de cero porque el flujo es siempre positivo. |
| `07_descomposicion_STL_NEE.png` ★ | La componente estacional domina sobre la tendencia: el balance de carbono lo manda el ciclo anual. Con tres años no se puede afirmar nada sobre tendencia a largo plazo. | Descomposición STL en tres paneles apilados —tendencia, estacionalidad y residuo— que suman la serie original; comparar sus amplitudes es lo que dice cuál manda. |
| `08_descomposicion_STL_FCH4.png` | Misma conclusión para el metano, con una estacionalidad aún más marcada. | Igual que la anterior, aplicada a la serie de FCH₄. |
| `09_climatologia_dia_del_anio.png` | **Es la curva que usa el simulador** para los drivers que el usuario no fija, así que condiciona toda simulación de escenarios. | Valor medio de cada driver para cada día del año, promediando los tres años con ventana circular de 15 días. El eje X es el día del año, no una fecha. |
| `10_estacionalidad_mensual_boxplots.png` | La misma señal por meses, mostrando la dispersión interanual que la climatología promedia. | Un diagrama de caja por mes con los valores de los tres años. |
| `11_distribucion_clase_balance_carbono.png` | Hace visible el desbalance ≈ 32/68 que obliga a usar F1 y AUC en vez de accuracy. | Recuento de días por clase, derivada del signo del NEE observado. |
| `13_qqplots_normalidad.png` | La comprobación visual de T2: desviaciones claras en las colas. | Gráficos cuantil-cuantil de cada serie frente a la normal teórica; si los datos fueran normales, los puntos caerían sobre la diagonal. |
| `14_perfil_termico_suelo_TS1_TS5.png` | La amplitud se amortigua con la profundidad mientras la media se mantiene: es el argumento gráfico para promediar las cinco sondas. | Las cinco series de temperatura de suelo superpuestas frente al tiempo, en °C. |

> No existe una figura 12: la numeración salta de 11 a 13. No falta nada, es un
> hueco heredado en el nombrado.

### 3.2 Entrenamiento — `ml/training/figuras/` · `python -m ml.training.run_training`

| Fichero | Interpretación | Explicabilidad |
|---|---|---|
| `comparativa_regresion_clasificacion.png` | Hace visible que no hay un modelo que domine en ambos ejes a la vez. | Sitúa cada modelo en un plano con una métrica de regresión en un eje y una de clasificación en el otro. |
| `matrices_confusion.png` ★ | Permite comparar **cómo** se equivoca cada modelo, no solo cuánto. Con 32 % de sumidero, un clasificador que respondiera «fuente» siempre ya lograría ≈ 68 % de accuracy sin aprender nada. Para restauración el error caro son los falsos positivos: sobreestiman el beneficio de una intervención. | Una matriz por modelo sobre el mismo holdout de 2018; filas = clase real, columnas = predicha, diagonal = aciertos. Clase positiva = sumidero (NEE < 0). Debajo de cada matriz, F1 y recall de esa clase. |
| `curvas_roc_comparacion.png` | AUC de 0.855 (CNN-LSTM) a 0.923 (Stacking): las cinco curvas quedan muy juntas. El margen de 0.068 es estrecho, y por eso hay que ir a T9 antes de afirmar superioridad. | Las cinco curvas sobre unos mismos ejes: tasa de falsos positivos frente a tasa de verdaderos positivos, recorriendo todos los umbrales. La diagonal es el azar y el AUC es el área bajo cada curva. |
| `heatmap_comparativo_metricas.png` ★ | Muestra **orden relativo, no magnitud**: un verde intenso puede ser un R² mediocre si todos lo son, como ocurre con el NEE. Nunca se lee sola, siempre junto a T6. Su utilidad es ver si algún modelo domina en bloque o si cada uno gana en una columna distinta. | Una fila por modelo y una columna por métrica. Cada **columna** se normaliza min-max por separado; en las métricas de error el sentido se invierte antes de colorear para que verde signifique siempre «mejor». Los números de las celdas son los valores normalizados, no las métricas originales. |
| `cv_boxplots_estabilidad.png` ★ | La altura de cada caja es la inestabilidad del modelo entre periodos. El solapamiento se repite en casi todas las métricas: ese es el argumento de fondo para no elegir ganador comparando medias. | Un panel por métrica y, dentro de cada uno, un diagrama de caja por modelo con sus valores a lo largo de los folds. |
| `statistical_tests_nemenyi_cd.png` ★ | Los modelos unidos por una misma barra **no** son distinguibles al 5 %. Con CD = 2.728 sobre una escala que solo va de 1 a 5, casi cualquier par queda unido: es la forma gráfica de decir que cinco folds no bastan. Solo CNN-LSTM se despega de Random Forest y de XGBoost. | Los cinco modelos sobre un eje de rango medio (1 = mejor RMSE medio entre folds, 5 = peor). La diferencia crítica CD es la distancia mínima entre rangos para que la diferencia sea significativa; las barras unen los grupos que no la alcanzan. |
| `statistical_tests_nemenyi_pmatrix.png` | El detalle par a par: solo 2 de los 10 pares bajan de 0.05. | Mapa de calor con el p-valor del post-hoc de Nemenyi para cada par de modelos. |
| `statistical_tests_distribuciones.png` | El solapamiento visual es la versión gráfica de por qué casi ningún contraste resulta significativo. | Las muestras sobre las que operan las pruebas: los valores de RMSE y de F1 macro que cada modelo obtuvo en cada uno de los cinco folds. |

### 3.3 Generadas al vuelo en los informes — `app/reports/figures.py`

No se guardan en disco: se construyen al generar el PDF o el Word.

| Función | Interpretación | Explicabilidad |
|---|---|---|
| `flux_series_png` | La compresión de amplitud de la curva predicha frente a la observada es la firma visual del R² bajo del NEE. | Dos paneles apilados con observado y predicho superpuestos: arriba CO₂ en gC m⁻² d⁻¹ con la línea de cero, abajo CH₄ en nmol m⁻² s⁻¹. Es un backtest con las variables reales de cada día. |
| `confusion_png` | La misma lectura que `matrices_confusion.png`, restringida al modelo desplegado. | Matriz 2×2 con filas = clase real y columnas = predicha; clase positiva = sumidero. |
| `roc_png` | Capacidad de ordenar días por probabilidad de sumidero, independiente del umbral 0.5. | Curva ROC del modelo activo con su AUC en la leyenda y la diagonal del azar como referencia. |

---

## 4. Dónde vive cada bloque en el código

No están escritos en un documento aparte que se pueda desincronizar: la
aplicación y los informes los emiten junto al dato.

- `ui/interpret.py` — los helpers. `interpretacion` y `explicabilidad` son
  argumentos **keyword-only obligatorios**: una tabla sin las dos no llega a
  ejecutarse. Se renderizan en ese orden, en un bloque con filete de acento.
- `views/motor_ia.py` — EDA, entrenamiento, comparativa, CV, tuning y pruebas
  estadísticas (24 pares).
- `views/series.py`, `views/simulador.py`, `views/modelos.py` — pares
  **calculados sobre lo que hay en pantalla**, no fijos: el rango de fechas, el
  nivel freático o un reentrenamiento cambian las cifras y el texto las sigue.
- `app/reports/docx_report.py` y `pdf_report.py` — `_interp()` y `_expl()`, en
  el mismo orden que la interfaz.
- `app/reports/xlsx_report.py` — hoja `lectura` con el diccionario de unidades,
  convenios de signo y salvedades de cada hoja.

---

## 5. Salvedades que deben acompañar a cualquier figura en el artículo

1. **Un solo sitio y tres años.** DE-Zrk (Zarnekow), 2016–2018, n = 1096 días.
   Los resultados describen esta turbera en este periodo; extrapolar a otra
   exige reentrenar y volver a validar.
2. **El periodo no es una elección libre.** 2013–2015 se descarta porque el WTD
   está 100 % vacío —también su versión rellenada— y es la variable de entrada
   del simulador.
3. **Los datos ya vienen procesados.** FLUXNET-CH4 entrega las series con
   gap-filling propio (ANNOPTLM); no son medidas crudas de torre y el proyecto
   no implementa relleno propio.
4. **Falta la vegetación.** Sin NDVI ni LAI, y el sitio tampoco midió humedad de
   suelo. Es la causa directa del R² bajo del NEE y hay que decirlo al presentar
   T6.
5. **Las diferencias entre modelos no son significativas** salvo en 3 de 24
   contrastes, y ninguno de esos tres implica al modelo desplegado (T9). La
   elección de Stacking se justifica por criterio combinado y estabilidad, no por
   superioridad demostrada.
6. **El modelo evaluado y el desplegado no son la misma fila.** Las pruebas
   estadísticas se hicieron sobre los 5 modelos base; el desplegado es
   `Stacking Ensemble (tuned)`, un refinamiento posterior del ganador ya
   seleccionado. Es defendible, pero hay que declararlo.
7. **Las curvas de respuesta son sensibilidad, no predicción.** Mueven el nivel
   freático con el resto de variables congeladas en su climatología; el
   rehumedecimiento real también cambia vegetación y temperatura de suelo.
8. **El compromiso CO₂/CH₄ depende del horizonte.** El simulador suma ambos
   gases con GWP100 = 27 (IPCC AR6, metano biogénico); con GWP20 (≈ 80) el peso
   del metano se triplica y el óptimo se desplaza. Es un supuesto declarado del
   lector, no una salida del modelo.
