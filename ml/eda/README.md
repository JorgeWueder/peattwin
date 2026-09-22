# EDA — DE-Zrk (Zarnekow)

Análisis exploratorio del dataset procesado por el ETL
(`data/processed/DE-Zrk_flux_daily_2016-2018.parquet`, 1096 días 2016–2018).

```bash
python -m ml.eda.run_eda        # regenera figuras, tablas y el resumen
```

## Salidas

| Ruta | Contenido |
|---|---|
| `EDA_DE-Zrk_resumen.md` | **Resumen con interpretación de cada hallazgo** (para el informe) |
| `figuras/*.png` | 13 figuras con nombre descriptivo y prefijo numérico |
| `tablas/*.csv` | Descriptivos, outliers, correlaciones (Pearson/Spearman), balance por mes, normalidad |
| `DE-Zrk_winsorized_k3.parquet` | Observaciones con las variables clave recortadas a 3·IQR (análisis de robustez; el dataset canónico **no** se toca) |

## Contenido del análisis

1. **Descriptivos** (media, mediana, desv. estándar, asimetría, curtosis de exceso) de
   `NEE_F_ANNOPTLM`, `FCH4_F_ANNOPTLM`, `WTD` y el perfil `TS_1..TS_5`.
2. **Outliers** — regla de Tukey (IQR): marcado a 1.5·IQR, extremo a 3.0·IQR.
   Tratamiento: no se eliminan filas; se entrega versión winsorizada a 3·IQR.
3. **Correlación** — heatmaps Pearson y Spearman (objetivos + WTD + perfil TS + meteo).
4. **Series temporales** — NEE y FCH4 diarios + media móvil 30 d, descomposición
   **STL** (period = 365), climatología por día del año, boxplots mensuales.
5. **`clase_balance_carbono`** — balance de clases con verificación del recuento
   esperado **353 sumidero / 743 fuente** (→ COINCIDE).
6. **Normalidad** — Shapiro-Wilk (α = 0.05) sobre variables clave y sobre los
   residuos STL, con D'Agostino-Pearson como contraste + QQ-plots.

## Limitación registrada

**DE-Zrk no midió humedad del suelo.** El producto FLUXNET-CH4 del sitio no
contiene ninguna variable SWC / `soil_water_content`; no se incluye en el EDA y
queda `NULL` en la tabla `flux_observations`. La hidrología se representa solo con
`WTD` (nivel freático) y `P_F` (precipitación).
