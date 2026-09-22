# EDA - DE-Zrk (Zarnekow) · flujos de carbono diarios 2016-2018

Fuente: `FLX_DE-Zrk_FLUXNET-CH4_DD_2013-2018_1-1.csv` (SHA-256 `a2b8d6464ccc20c3...`,
2191 filas crudas) -> ETL -> `DE-Zrk_flux_daily_2016-2018.parquet` (1096 filas,
2016-01-01 .. 2018-12-31, serie diaria continua).

Reproducible con: `python -m ml.eda.run_eda`. Figuras en `ml/eda/figuras/`,
tablas en `ml/eda/tablas/`.

## Limitacion del dataset (importante para el informe)

**El sitio DE-Zrk no midio humedad del suelo.** El producto FLUXNET-CH4 de este
sitio no incluye ninguna variable de contenido de agua del suelo
(`soil_water_content` / SWC). No se incluye en este EDA y en la tabla
`flux_observations` queda `NULL`. La unica variable hidrologica disponible es el
nivel freatico (`WTD`), que ademas es el driver que usa el simulador de
escenarios del dashboard. Cualquier analisis de "humedad" debe apoyarse en `WTD`
y en la precipitacion (`P_F`), no en SWC.

Otras limitaciones heredadas del ETL: 2013-2015 descartados (WTD 100 % vacio);
14 dias de 2016-2018 con WTD/TS interpolados linealmente (<= 7 d), marcados con
`*_gapfilled_by_etl`.

## 1. Estadisticos descriptivos

| variable | n | media | mediana | desv_std | min | max | q25 | q75 | IQR | asimetria | curtosis_exceso |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEE_F_ANNOPTLM | 1096 | 0.052 | 0.284 | 1.079 | -4.646 | 3.739 | -0.306 | 0.637 | 0.943 | -1.214 | 2.613 |
| FCH4_F_ANNOPTLM | 1096 | 80.385 | 22.941 | 99.645 | 0.412 | 690.199 | 9.690 | 150.544 | 140.854 | 1.565 | 2.764 |
| WTD | 1096 | 0.226 | 0.205 | 0.214 | -0.119 | 0.798 | 0.090 | 0.318 | 0.228 | 0.589 | -0.230 |
| TS_1 | 1096 | 10.956 | 10.284 | 5.461 | 3.124 | 22.116 | 5.394 | 16.315 | 10.921 | 0.179 | -1.454 |
| TS_2 | 1096 | 10.905 | 10.233 | 4.929 | 3.686 | 20.196 | 5.876 | 15.700 | 9.824 | 0.160 | -1.485 |
| TS_3 | 1096 | 11.010 | 10.357 | 4.434 | 4.462 | 18.795 | 6.465 | 15.350 | 8.886 | 0.144 | -1.500 |
| TS_4 | 1096 | 10.970 | 10.357 | 3.980 | 5.058 | 17.625 | 6.931 | 14.837 | 7.905 | 0.140 | -1.505 |
| TS_5 | 1096 | 10.930 | 10.385 | 3.564 | 5.595 | 16.650 | 7.364 | 14.461 | 7.097 | 0.136 | -1.503 |

*Figuras:* `01_histogramas_kde_variables_clave.png`, `14_perfil_termico_suelo_TS1_TS5.png`

**Interpretacion**
- **NEE_F_ANNOPTLM**: media = +0.05 gC m-2 d-1 (practicamente
  neutra en promedio diario) pero mediana = +0.28 y solo
  32.2 % de dias con captura -> **el sitio es fuente de CO2 la
  mayoria de los dias**, con episodios intensos de captura en verano que tiran de
  la media hacia cero. Asimetria = -1.21 (cola izquierda) y
  curtosis de exceso = +2.61 (leptocurtica): coherente con
  ese regimen estacional bimodal.
- **FCH4_F_ANNOPTLM**: media = 80.39 nmol m-2 s-1, mediana =
  22.94, **asimetria = +1.57** (cola derecha
  marcada) y curtosis de exceso = +2.76. Tipico de un
  flujo estrictamente positivo con pulsos estivales: convendra transformar
  (log / sqrt) antes de modelos que asuman normalidad de residuos.
- **WTD**: media = +0.226 m, rango [-0.12, +0.80] m.
  Predominantemente **positivo** = lamina de agua por encima de la superficie:
  confirma que Zarnekow es un fen **rehumedecido/inundado**.
- **Perfil TS_1..TS_5**: la media es casi identica (~10.9 C) pero la **desviacion
  estandar decrece monotonamente con la profundidad**
  (5.46 C a 0.05 m -> 3.56 C
  a ~0.48 m): amortiguamiento termico clasico. Para modelar basta 1-2
  profundidades (el resto es redundante, ver correlacion).

## 2. Deteccion y tratamiento de outliers

**Criterio:** regla de Tukey sobre el IQR. "Marcado" fuera de
[Q1 - 1.5·IQR, Q3 + 1.5·IQR]; "extremo" fuera de [Q1 - 3.0·IQR, Q3 + 3.0·IQR].

| variable | Q1 | Q3 | IQR | lim_inf_1.5 | lim_sup_1.5 | n_out_1.5 | pct_1.5 | n_out_3.0 | n_winsorizados_k3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEE_F_ANNOPTLM | -0.31 | 0.64 | 0.94 | -1.72 | 2.05 | 97 | 8.85 | 21 | 21 |
| FCH4_F_ANNOPTLM | 9.69 | 150.54 | 140.85 | -201.59 | 361.82 | 15 | 1.37 | 2 | 2 |
| WTD | 0.09 | 0.32 | 0.23 | -0.25 | 0.66 | 52 | 4.74 | 0 | 0 |
| TS_1 | 5.39 | 16.31 | 10.92 | -10.99 | 32.70 | 0 | 0 | 0 | 0 |
| TS_2 | 5.88 | 15.70 | 9.82 | -8.86 | 30.44 | 0 | 0 | 0 | 0 |
| TS_3 | 6.46 | 15.35 | 8.89 | -6.86 | 28.68 | 0 | 0 | 0 | 0 |
| TS_4 | 6.93 | 14.84 | 7.91 | -4.93 | 26.70 | 0 | 0 | 0 | 0 |
| TS_5 | 7.36 | 14.46 | 7.10 | -3.28 | 25.11 | 0 | 0 | 0 | 0 |

*Figura:* `02_boxplots_outliers_iqr.png`

**Tratamiento aplicado**
- **No se elimina ninguna fila.** Los valores atipicos de NEE y FCH4 son senal
  geofisica real (pulsos de emision en olas de calor, episodios de respiracion
  tras lluvia). Eliminarlos sesgaria los balances anuales de C.
- Se entrega una version **winsorizada al limite 3.0·IQR**
  (`ml/eda/DE-Zrk_winsorized_k3.parquet`) para analisis de robustez / modelos
  sensibles a colas. Nº de valores recortados por variable en la tabla anterior
  (columna `n_winsorizados_k3`).
- Los outliers de WTD/TS se concentran en los 14 dias interpolados y en
  transiciones estacionales; se conservan.

## 3. Matriz de correlacion

*Figuras:* `03_correlacion_pearson_heatmap.png`, `04_correlacion_spearman_heatmap.png`
· tablas `03_correlacion_pearson.csv` / `_spearman.csv`

**Interpretacion**
- Predictores mas correlacionados con **NEE**: SW_IN_F = -0.60, NETRAD_F = -0.57, VPD_F = -0.45, TS_1 = -0.43, TS_2 = -0.41.
- Predictores mas correlacionados con **FCH4**: TS_1 = +0.80, TA_F = +0.79, TS_2 = +0.77, NETRAD_F = +0.72, VPD_F = +0.72.
- **TS_1..TS_5 estan casi colineales entre si** (r > 0.95): se usa TS_1 (y TS_2)
  como feature y el resto queda solo como contexto -> justifica la decision del
  ETL de no meter TS_3..TS_5 en la matriz ML.
- La radiacion (`SW_IN_F`, `NETRAD_F`), `TA_F` y `VPD_F` estan muy
  intercorrelacionadas (bloque meteo-energetico): hay redundancia que un modelo
  regularizado o un arbol gestionaran bien, pero conviene tenerlo en cuenta para
  interpretar importancias.

## 4. Series temporales y estacionalidad

*Figuras:* `05_serie_temporal_NEE_CO2.png`, `06_serie_temporal_FCH4_CH4.png`,
`07_descomposicion_STL_NEE.png`, `08_descomposicion_STL_FCH4.png`,
`09_climatologia_dia_del_anio.png`, `10_estacionalidad_mensual_boxplots.png`

**Interpretacion**
- **NEE**: fuerte ciclo anual: valores negativos (captura) en el pico de verano y
  positivos (emision) en invierno. La componente estacional STL domina sobre la
  tendencia; la tendencia interanual es debil en solo 3 anios (no concluir
  cambios de regimen con esta ventana).
- **FCH4**: estacionalidad muy marcada con maximo en verano (temperatura del
  suelo) y minimo invernal; la serie nunca baja de ~0 (emision permanente,
  propia de un fen inundado). Amplitud estacional creciente los anios mas
  calidos.
- La climatologia por dia del anio resume ambos patrones y sera la base para las
  features estacionales del modelo (`doy_sin/cos`, `month_sin/cos`).

## 5. Distribucion de `clase_balance_carbono`

Verificacion del recuento esperado (353 sumidero / 743 fuente):
**COINCIDE**
(sumidero = 353, fuente = 743,
32.2 % de dias sumidero).

*Figura:* `11_distribucion_clase_balance_carbono.png` · tabla
`05_clase_balance_por_mes.csv`

**Interpretacion**
- Problema de clasificacion **desbalanceado ~1:2**. Hay que usar metricas
  robustas al desbalance (F1, AUC-ROC, balanced accuracy) y considerar
  `class_weight` / remuestreo.
- El desglose mensual muestra que la clase "sumidero" se concentra en
  mayo-agosto; fuera de esa ventana casi todos los dias son "fuente". Un modelo
  con las features estacionales ya captura gran parte de la senal.

## 6. Pruebas de normalidad (Shapiro-Wilk, alpha = 0.05)

| serie | n | shapiro_W | shapiro_p | normal_shapiro_5pct | dagostino_K2 | dagostino_p | normal_dagostino_5pct | asimetria | curtosis_exceso |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEE_F_ANNOPTLM | 1096 | 0.8983 | 0.0000 | no | 246.2222 | 0.0000 | no | -1.2143 | 2.6125 |
| FCH4_F_ANNOPTLM | 1096 | 0.7555 | 0.0000 | no | 320.9237 | 0.0000 | no | 1.5652 | 2.7641 |
| WTD | 1096 | 0.9544 | 0.0000 | no | 58.3795 | 0.0000 | no | 0.5893 | -0.2300 |
| TS_1 | 1096 | 0.9078 | 0.0000 | no | 9440.1707 | 0 | no | 0.1795 | -1.4539 |
| TS_2 | 1096 | 0.9035 | 0.0000 | no | 8210.3604 | 0 | no | 0.1602 | -1.4852 |
| TS_3 | 1096 | 0.9016 | 0.0000 | no | 7781.5896 | 0 | no | 0.1437 | -1.5002 |
| TS_4 | 1096 | 0.9005 | 0.0000 | no | 7656.0366 | 0 | no | 0.1401 | -1.5052 |
| TS_5 | 1096 | 0.9015 | 0.0000 | no | 7699.4643 | 0 | no | 0.1363 | -1.5034 |
| residuo_STL_NEE | 1096 | 0.6298 | 0.0000 | no | 231.1558 | 0.0000 | no | 0.1849 | 12.1513 |
| residuo_STL_FCH4 | 1096 | 0.5401 | 0.0000 | no | 561.0276 | 0.0000 | no | -2.1430 | 10.2134 |

**Interpretacion**
- **Ninguna** de las variables clave supera Shapiro-Wilk: se rechaza normalidad
  (p < 0.05) en todos los casos. D'Agostino-Pearson coincide.
- Los **residuos STL** tampoco son normales. El residuo de NEE es casi simetrico
  (asimetria +0.18) pero con **colas muy pesadas**
  (curtosis de exceso +12.15); el de FCH4 es ademas
  fuertemente asimetrico (-2.14) y leptocurtico
  (+10.21). Shapiro-Wilk p = 2.2e-43
  y 1.1e-46 respectivamente.
- **Consecuencia para el paso de pruebas estadisticas:** usar contrastes **no
  parametricos** (Wilcoxon / Mann-Whitney, Kruskal-Wallis, correlacion de
  Spearman) o transformar las variables (log1p en FCH4). Para comparar modelos,
  preferir tests no parametricos sobre los errores (p. ej. Wilcoxon de rangos
  con signo, o Diebold-Mariano) en lugar de un t-test.
- Matiz: con n ~ 1100 Shapiro-Wilk es muy potente y rechaza con desviaciones
  pequenas. Aun asi, las figuras QQ (`13_qqplots_normalidad.png`) confirman
  desviaciones **no** despreciables: colas pesadas en el residuo de NEE y
  asimetria marcada en el de FCH4. La conclusion practica (usar no parametricos)
  se mantiene.

## Indice de figuras

| Archivo | Contenido |
|---|---|
| `01_histogramas_kde_variables_clave.png` | Histograma + KDE de NEE, FCH4, WTD, TS_1..TS_5 |
| `02_boxplots_outliers_iqr.png` | Boxplots estandarizados (bigotes 1.5·IQR) |
| `03_correlacion_pearson_heatmap.png` | Heatmap de correlacion de Pearson |
| `04_correlacion_spearman_heatmap.png` | Heatmap de correlacion de Spearman |
| `05_serie_temporal_NEE_CO2.png` | Serie diaria de NEE + media movil 30 d |
| `06_serie_temporal_FCH4_CH4.png` | Serie diaria de FCH4 + media movil 30 d |
| `07_descomposicion_STL_NEE.png` | STL de NEE (observado/tendencia/estacional/residuo) |
| `08_descomposicion_STL_FCH4.png` | STL de FCH4 |
| `09_climatologia_dia_del_anio.png` | Climatologia por dia del anio (NEE y FCH4) |
| `10_estacionalidad_mensual_boxplots.png` | Boxplots mensuales de NEE y FCH4 |
| `11_distribucion_clase_balance_carbono.png` | Balance de clases + fraccion por mes |
| `13_qqplots_normalidad.png` | QQ-plots vs normal (variables + residuos STL) |
| `14_perfil_termico_suelo_TS1_TS5.png` | Amortiguamiento termico con la profundidad |

## Tablas (CSV en `ml/eda/tablas/`)

`01_descriptivos.csv` · `02_outliers_iqr.csv` · `03_correlacion_pearson.csv` ·
`03_correlacion_spearman.csv` · `05_clase_balance_por_mes.csv` ·
`06_normalidad.csv`

## Artefacto de tratamiento de outliers

`ml/eda/DE-Zrk_winsorized_k3.parquet` - copia de las observaciones con las
variables clave recortadas al limite 3.0·IQR (para analisis de robustez; el
dataset canonico NO se modifica).
