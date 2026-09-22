# Resumen de entrenamiento - flujos de carbono DE-Zrk

Evaluacion: holdout temporal train = 2016-2017 (702 dias) -> test = 2018 (365 dias).
32 features (TS_1..TS_5 sustituidas por TS_promedio).

## Verificacion: drivers de radiacion en el conjunto de features

| feature | ¿incluida? | rol |
|---|---|---|
| `SW_IN_F` (radiacion onda corta entrante) | SI | control primario de GPP |
| `VPD_F` (deficit de presion de vapor) | SI | control de la conductancia estomatica |
| `NETRAD_F` (radiacion neta) | SI | balance energetico |
| `PPFD_IN_F` (radiacion fotosinteticamente activa) | NO (excluida) | 65 dias vacios en ene-mar 2018; se usa SW_IN_F como proxy |

**Resultado: SW_IN_F y VPD_F (y NETRAD_F) YA estaban incluidos** desde el ETL
(`METEO_FEATURES`). No hay un driver de radiacion "perdido" que explique el bajo
R2 de NEE.

## Comparativa de modelos (holdout 2018)

| model | train_time_s | nee_rmse | nee_r2 | fch4_rmse | fch4_r2 | accuracy | f1_macro | recall_sumidero | f1_sumidero | auc | combined_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 2.128 | 1.285 | 0.206 | 45.301 | 0.831 | 0.822 | 0.789 | 0.650 | 0.706 | 0.911 | 0.906 |
| Gradient Boosting (XGBoost) | 1.163 | 1.324 | 0.157 | 43.545 | 0.844 | 0.825 | 0.803 | 0.750 | 0.738 | 0.902 | 0.861 |
| SVR / SVM | 0.139 | 1.275 | 0.219 | 58.325 | 0.720 | 0.805 | 0.748 | 0.500 | 0.628 | 0.914 | 0.771 |
| Stacking Ensemble | 17.628 | 1.277 | 0.216 | 44.500 | 0.837 | 0.844 | 0.827 | 0.808 | 0.773 | 0.923 | 0.937 |
| CNN-LSTM | 13.022 | 1.291 | 0.199 | 80.883 | 0.461 | 0.805 | 0.799 | 0.950 | 0.763 | 0.855 | 0.413 |

Ganador (criterio 0.5*reg_score_norm + 0.5*(0.5*F1_macro + 0.5*AUC)): **Stacking Ensemble**.

## Por que el R2 de NEE (0.16-0.22) es mucho menor que el de FCH4 (0.72-0.84 en RF/XGB/SVR/Stacking; el CNN-LSTM baja a 0.46 por falta de datos para una red profunda)

No es un problema de features de radiacion faltantes (estan todas). El NEE en
DE-Zrk es **intrinsecamente mas dificil** de predecir con drivers
meteorologicos/hidrologicos solos:

1. **NEE = GPP - RECO**: es la pequena diferencia entre dos flujos grandes de
   signo opuesto (fotosintesis y respiracion) que casi se cancelan. La relacion
   senal/ruido del residuo es baja: errores modestos en GPP o en RECO se
   amplifican en NEE. FCH4, en cambio, es un flujo unico y estrictamente
   positivo.
2. **Falta el estado de la vegetacion.** La GPP depende de la fenologia y de la
   cantidad/actividad de hoja verde (rebrote primaveral, siega, senescencia,
   estres hidrico). Este dataset FLUXNET-CH4 **no incluye NDVI, LAI, EVI ni
   fraccion de cobertura verde**. La radiacion (SW_IN_F, NETRAD_F, VPD_F)
   explica la *envolvente estacional* de la GPP, pero no su variacion dia a dia
   por cambios en la capacidad fotosintetica del dosel.
3. **FCH4 esta bien restringido por lo que si hay**: temperatura del suelo
   (TS_promedio) y nivel freatico (WTD) controlan la produccion y el transporte
   de metano, y ambos estan medidos y con buena cobertura -> R2 alto.
4. **Media de NEE ~ 0** (0.05 gC m-2 d-1) con desviacion ~ 1.08: el "modelo
   nulo" (predecir la media) ya captura la escala, de modo que queda poca
   varianza estructurada que un modelo pueda ganar sin informacion de
   vegetacion; el R2, que se mide contra ese modelo nulo, castiga mucho.

**Consecuencia para el informe / trabajo futuro:** para subir el R2 de NEE haria
falta incorporar indices de vegetacion de satelite (MODIS/Sentinel-2: NDVI, LAI,
EVI) o modelar por separado GPP y RECO (particion de flujo) como objetivos
intermedios. Con los drivers disponibles, R2 ~ 0.2 para NEE diario es un
resultado esperable y coherente con la literatura de flujos en turberas.

*(generado por `python -m ml.training.run_training`)*
