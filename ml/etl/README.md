# ETL — DE-Zrk (Zarnekow) · flujos de carbono diarios

Pipeline de preparación de datos para el gemelo digital, a partir del producto
**FLUXNET-CH4 2015** del sitio **DE-Zrk / Zarnekow** (Alemania): turbera tipo
*fen* rehumedecida hidrológicamente en 2004–2005. Resolución diaria, 2013–2018,
2191 filas.

```
ml/etl/
  config.py       # parámetros: rutas, sitio, periodo, columnas, split, mapeos a BD
  run_etl.py      # pasos 1–7 -> 2 parquet + 1 JSON de metadatos
  load_to_db.py   # script separado: parquet de observaciones -> tabla flux_observations
```

## Ejecución (desde la raíz del repo, con el venv de `ml/` activo)

```bash
python -m ml.etl.run_etl
python -m ml.etl.load_to_db            # requiere el esquema migrado (alembic upgrade head)
python -m ml.etl.load_to_db --dry-run  # inspección sin escribir
```

## Salidas en `data/processed/`

| Archivo | Filas | Uso |
|---|---|---|
| `DE-Zrk_flux_daily_2016-2018.parquet` | 1096 | Observaciones diarias limpias → `flux_observations` |
| `DE-Zrk_ml_dataset_2016-2018.parquet` | 1067 | Matriz ML: objetivos + 33 features + columna `split` |
| `DE-Zrk_etl_metadata.json` | — | Fuente + SHA-256, conteos por paso, log de limpieza, supuestos, balance de clases |

## Decisiones clave (evidencia para el informe)

| # | Decisión | Justificación verificada |
|---|---|---|
| 1 | `-9999 → NaN` | Centinela FLUXNET; 0 residuales tras la conversión |
| 2 | Serie diaria continua | 2191 filas = 2191 días, **0 huecos** de calendario 2013–2018 |
| 3 | Objetivos regresión = `NEE_F_ANNOPTLM`, `FCH4_F_ANNOPTLM` | Ya *gap-filled* al 100 % por el sitio (red neuronal ANNOPTLM) → sin gap-filling propio |
| 4 | Clasificación `clase_balance_carbono` | `NEE_F_ANNOPTLM < 0` → **sumidero**; `≥ 0` → **fuente**. Desbalance ~33 % / ~67 % |
| 5 | **Periodo 2016–2018** | `WTD` (y `WTD_F`) **0 % de cobertura** en 2013/2014/2015; 100/96/100 % en 2016/17/18 |
| 6 | `WTD` = entrada obligatoria | El simulador de escenarios del dashboard (Paso 10) perturba `WTD` para generar los escenarios de restauración; sin `WTD` no hay eje de escenario |
| 7 | Interpolación lineal ≤ 7 días en `WTD`/`TS_1`/`TS_2` | Solo 14 días afectados en 2016–2018; filas marcadas con `*_gapfilled_by_etl` |
| 8 | Exclusión de columnas de flujo como features | `NEE*`, `H*`, `LE*`, `FCH4*`, `GPP_*`, `RECO_*`, `USTAR` → fuga de información |
| 9 | Exclusión de `PPFD_IN_F` | 65 días vacíos en ene–mar 2018; se usa `SW_IN_F` como proxy de radiación |
| 10 | Ventanas móviles con ventana completa | Se descartan los 29 primeros días de 2016 en la matriz ML (1096 → 1067) |
| 11 | **Split temporal** train = 2016–2017, test = 2018 | Nunca aleatorio; imita el pronóstico operativo y respeta la autocorrelación |
| 12 | `soil_water_content` → NULL | No existe columna equivalente en este dataset |
| 13 | `WTD` m → cm al cargar en BD | La columna `water_table_depth` está documentada en cm |
| 14 | `timestamp` diario anclado a 12:00 UTC en BD | `timestamp::date` devuelve el día correcto en cualquier zona de sesión |

## Features generadas (33)

- **Drivers (13):** `WTD`, `TS_1`, `TS_2`, `TA_F`, `P_F`, `VPD_F`, `SW_IN_F`, `RH_F`, `WS_F`, `PA_F`, `NETRAD_F`, `LW_IN_F`, `G_F`
- **Lags (6):** `WTD_lag{1,3,7}`, `TS_1_lag{1,3,7}`
- **Móviles (8):** `WTD_roll{7,30}_mean`, `TS_1_roll{7,30}_mean`, `TA_F_roll{7,30}_mean`, `P_F_roll{7,30}_sum`
- **Estacionales (6):** `doy`, `month`, `doy_sin`, `doy_cos`, `month_sin`, `month_cos`

## Mapeo a `flux_observations`

| Columna BD | Origen | Nota |
|---|---|---|
| `nee_co2` | `NEE_F_ANNOPTLM` | gC m⁻² d⁻¹ |
| `fch4` | `FCH4_F_ANNOPTLM` | nmol CH₄ m⁻² s⁻¹ |
| `water_table_depth` | `WTD × 100` | m → cm |
| `soil_temp` | `TS_1` | °C, 0.05 m |
| `air_temp` | `TA_F` | °C |
| `precipitation` | `P_F` | mm d⁻¹ |
| `soil_water_content` | — | NULL (no disponible) |
| `carbon_balance_class` | signo de `NEE_F_ANNOPTLM` | `SINK` / `SOURCE` |
| `quality_flag` | `FCH4_F_ANNOPTLM_QC` | entero |

`load_to_db.py` es **idempotente**: *upsert* por la restricción única
`(site_id, timestamp)`. Crea el sitio `DE-Zrk` si no existe.
