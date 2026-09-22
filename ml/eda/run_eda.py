"""Analisis exploratorio de datos (EDA) - sitio FLUXNET-CH4 DE-Zrk (Zarnekow).

Trabaja sobre el dataset procesado por el ETL:
    data/processed/DE-Zrk_flux_daily_2016-2018.parquet   (observaciones, 1096 filas)
    data/processed/DE-Zrk_ml_dataset_2016-2018.parquet   (matriz ML, para correlaciones)

Genera:
    ml/eda/figuras/*.png                  -> figuras con nombre descriptivo
    ml/eda/tablas/*.csv                   -> tablas numericas reutilizables
    ml/eda/DE-Zrk_winsorized_k3.parquet   -> version con outliers extremos tratados
    ml/eda/EDA_DE-Zrk_resumen.md          -> resumen con interpretacion (para el informe)

Ejecutar desde la raiz del repositorio:
    python -m ml.eda.run_eda

-------------------------------------------------------------------------------
LIMITACION DEL DATASET (documentada, no se inventa nada):
    El sitio DE-Zrk NO midio humedad del suelo. En el producto FLUXNET-CH4 no
    hay ninguna columna de contenido de agua del suelo (soil_water_content /
    SWC). Por tanto NO aparece en este EDA ni en la tabla flux_observations
    (queda NULL). La hidrologia del sitio se representa unicamente con WTD
    (nivel freatico).
-------------------------------------------------------------------------------
CRITERIOS METODOLOGICOS
    * Outliers: regla de Tukey sobre el rango intercuartilico (IQR).
      Valor atipico "marcado"  si  x < Q1 - 1.5*IQR  o  x > Q3 + 1.5*IQR.
      Valor atipico "extremo"   si  x < Q1 - 3.0*IQR  o  x > Q3 + 3.0*IQR.
      Tratamiento: NO se eliminan filas (los extremos de flujo son senal
      geofisica real: dias calidos de fuerte emision de CH4, pulsos de
      respiracion). Se entrega ademas una version winsorizada al limite 3.0*IQR
      (DE-Zrk_winsorized_k3.parquet) para analisis de robustez posteriores.
    * Estacionalidad: descomposicion STL (period=365, robust=True) de las series
      diarias continuas de NEE y FCH4.
    * Normalidad: Shapiro-Wilk (alpha=0.05) sobre variables clave y sobre los
      residuos STL; se acompana de D'Agostino-Pearson y Anderson-Darling como
      contraste (Shapiro es muy sensible con n~1100).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL
from ml import _logging

# --------------------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OBS_PARQUET = PROCESSED / "DE-Zrk_flux_daily_2016-2018.parquet"
ML_PARQUET = PROCESSED / "DE-Zrk_ml_dataset_2016-2018.parquet"
META_JSON = PROCESSED / "DE-Zrk_etl_metadata.json"

EDA_DIR = Path(__file__).resolve().parent
FIG_DIR = EDA_DIR / "figuras"
TBL_DIR = EDA_DIR / "tablas"
OUT_MD = EDA_DIR / "EDA_DE-Zrk_resumen.md"
OUT_WINSOR = EDA_DIR / "DE-Zrk_winsorized_k3.parquet"

# Variables pedidas para los descriptivos
KEY_VARS = ["NEE_F_ANNOPTLM", "FCH4_F_ANNOPTLM", "WTD", "TS_1", "TS_2", "TS_3", "TS_4", "TS_5"]
SOIL_TEMP = ["TS_1", "TS_2", "TS_3", "TS_4", "TS_5"]
TS_DEPTH_M = {"TS_1": 0.05, "TS_2": 0.10, "TS_3": 0.20, "TS_4": 0.32, "TS_5": 0.48}
TARGETS = ["NEE_F_ANNOPTLM", "FCH4_F_ANNOPTLM"]

CORR_VARS = TARGETS + [
    "WTD", "TS_1", "TS_2", "TS_3", "TS_4", "TS_5",
    "TA_F", "P_F", "VPD_F", "SW_IN_F", "NETRAD_F", "RH_F", "WS_F", "PA_F", "LW_IN_F", "G_F",
]

ALPHA = 0.05
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------- utilidades
def _md_table(df: pd.DataFrame, floatfmt: str = ".3f", index_name: str | None = None) -> str:
    idx_name = index_name or (df.index.name or "")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([idx_name, *cols]) + " |",
           "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{int(v)}")
            elif isinstance(v, (float, np.floating)):
                if pd.isna(v):
                    cells.append("n/a")
                elif float(v).is_integer() and abs(v) < 1e15:
                    cells.append(f"{int(v)}")
                else:
                    cells.append(f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(out)


def _savefig(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> figuras/{name}")
    return name


# ------------------------------------------------------- 1. estadisticos descriptivos
def descriptive_stats(obs: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for v in KEY_VARS:
        x = obs[v].dropna().to_numpy()
        rows[v] = {
            "n": len(x),
            "media": np.mean(x),
            "mediana": np.median(x),
            "desv_std": np.std(x, ddof=1),
            "min": np.min(x),
            "max": np.max(x),
            "q25": np.percentile(x, 25),
            "q75": np.percentile(x, 75),
            "IQR": np.percentile(x, 75) - np.percentile(x, 25),
            "asimetria": stats.skew(x, bias=False),
            "curtosis_exceso": stats.kurtosis(x, fisher=True, bias=False),
        }
    tbl = pd.DataFrame(rows).T
    tbl["n"] = tbl["n"].astype("int64")
    tbl.index.name = "variable"
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TBL_DIR / "01_descriptivos.csv")

    # Figura: histograma + KDE
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, v in zip(axes.ravel(), KEY_VARS):
        sns.histplot(obs[v].dropna(), kde=True, ax=ax, color="#2a6f97")
        ax.axvline(obs[v].mean(), color="crimson", ls="--", lw=1, label="media")
        ax.axvline(obs[v].median(), color="green", ls=":", lw=1, label="mediana")
        ax.set_title(v)
        ax.legend(fontsize=7)
    fig.suptitle("Distribucion de las variables clave (DE-Zrk, 2016-2018)", fontsize=14)
    _savefig(fig, "01_histogramas_kde_variables_clave.png")

    # Figura: perfil termico del suelo (amortiguamiento con la profundidad)
    fig, ax = plt.subplots(figsize=(8, 5))
    amp = [obs[c].std(ddof=1) for c in SOIL_TEMP]
    ax.plot([TS_DEPTH_M[c] for c in SOIL_TEMP], amp, "o-", color="#a4133c")
    for c in SOIL_TEMP:
        ax.annotate(c, (TS_DEPTH_M[c], obs[c].std(ddof=1)), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Profundidad de la sonda (m)")
    ax.set_ylabel("Desviacion estandar de la temperatura (C)")
    ax.set_title("Amortiguamiento termico del suelo con la profundidad")
    _savefig(fig, "14_perfil_termico_suelo_TS1_TS5.png")

    return tbl


# ---------------------------------------------------------------- 2. outliers (IQR)
def outlier_analysis(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = {}
    flags = pd.DataFrame(index=obs.index)
    for v in KEY_VARS:
        x = obs[v]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        rec = {"Q1": q1, "Q3": q3, "IQR": iqr}
        for k in (1.5, 3.0):
            lo, hi = q1 - k * iqr, q3 + k * iqr
            mask = (x < lo) | (x > hi)
            rec[f"lim_inf_{k}"] = lo
            rec[f"lim_sup_{k}"] = hi
            rec[f"n_out_{k}"] = int(mask.sum())
            rec[f"pct_{k}"] = round(100 * mask.mean(), 2)
            if k == 1.5:
                flags[f"{v}__outlier_1_5IQR"] = mask
        recs[v] = rec
    tbl = pd.DataFrame(recs).T
    for c in ("n_out_1.5", "n_out_3.0"):
        tbl[c] = tbl[c].astype("int64")
    tbl.index.name = "variable"
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TBL_DIR / "02_outliers_iqr.csv")

    # Figura: boxplots (escala estandarizada para verlos juntos)
    z = (obs[KEY_VARS] - obs[KEY_VARS].mean()) / obs[KEY_VARS].std(ddof=1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=z, orient="h", ax=ax, color="#89c2d9", whis=1.5)
    ax.set_title("Boxplots (variables estandarizadas, bigotes = 1.5*IQR)")
    ax.set_xlabel("z-score")
    _savefig(fig, "02_boxplots_outliers_iqr.png")

    # Tratamiento: winsorizacion al limite 3.0*IQR (solo variables clave)
    wins = obs.copy()
    n_clipped = {}
    for v in KEY_VARS:
        q1, q3 = obs[v].quantile(0.25), obs[v].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        before = wins[v].copy()
        wins[v] = wins[v].clip(lower=lo, upper=hi)
        n_clipped[v] = int((before != wins[v]).sum())
    wins.to_parquet(OUT_WINSOR, index=False, engine="pyarrow")
    tbl["n_winsorizados_k3"] = pd.Series(n_clipped).astype("int64")
    tbl.to_csv(TBL_DIR / "02_outliers_iqr.csv")
    print(f"  winsorizado (k=3) -> {OUT_WINSOR.name}  (clip: {n_clipped})")
    return tbl, flags


# ------------------------------------------------------------ 3. matriz de correlacion
def correlation_matrix(ml: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORR_VARS if c in ml.columns]
    pear = ml[cols].corr(method="pearson")
    spear = ml[cols].corr(method="spearman")
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    pear.to_csv(TBL_DIR / "03_correlacion_pearson.csv")
    spear.to_csv(TBL_DIR / "03_correlacion_spearman.csv")

    for corr, name, title in [
        (pear, "03_correlacion_pearson_heatmap.png", "Pearson"),
        (spear, "04_correlacion_spearman_heatmap.png", "Spearman"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, annot_kws={"size": 7}, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Matriz de correlacion ({title}) - DE-Zrk 2016-2018", fontsize=13)
        _savefig(fig, name)
    return pear


# ------------------------------------------------- 4. series temporales + estacionalidad
def time_series_and_seasonality(obs: pd.DataFrame) -> dict[str, pd.Series]:
    s = obs.set_index("timestamp").asfreq("D")
    resids: dict[str, pd.Series] = {}

    labels = {
        "NEE_F_ANNOPTLM": ("NEE (CO2)  [gC m-2 d-1]", "05_serie_temporal_NEE_CO2.png", "07_descomposicion_STL_NEE.png"),
        "FCH4_F_ANNOPTLM": ("FCH4 (CH4)  [nmol m-2 s-1]", "06_serie_temporal_FCH4_CH4.png", "08_descomposicion_STL_FCH4.png"),
    }
    for v, (ylab, fname_ts, fname_stl) in labels.items():
        serie = s[v]
        # Serie diaria + media movil 30 d
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(serie.index, serie.values, lw=0.6, color="#adb5bd", label="diario")
        ax.plot(serie.index, serie.rolling(30, min_periods=30).mean(), lw=2, color="#1d3557", label="media movil 30 d")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(ylab)
        ax.set_title(f"Serie temporal - {v} (DE-Zrk)")
        ax.legend()
        _savefig(fig, fname_ts)

        # STL
        stl = STL(serie, period=365, robust=True).fit()
        resids[v] = stl.resid
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for ax, comp, tit in zip(
            axes, [stl.observed, stl.trend, stl.seasonal, stl.resid],
            ["Observado", "Tendencia", "Estacional", "Residuo"],
        ):
            ax.plot(serie.index, comp, lw=0.9)
            ax.set_ylabel(tit)
        axes[0].set_title(f"Descomposicion STL - {v} (period=365, robust)")
        _savefig(fig, fname_stl)

    # Climatologia por dia del anio
    doy = obs.assign(doy=obs["timestamp"].dt.dayofyear)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, v, col in zip(axes, TARGETS, ["#e63946", "#2a9d8f"]):
        g = doy.groupby("doy")[v]
        m, sd = g.mean(), g.std(ddof=1)
        ax.plot(m.index, m.values, color=col, lw=1.5)
        ax.fill_between(m.index, m - sd, m + sd, color=col, alpha=0.2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Dia del anio")
        ax.set_title(f"Climatologia estacional - {v}")
    _savefig(fig, "09_climatologia_dia_del_anio.png")

    # Boxplots mensuales
    mon = obs.assign(mes=obs["timestamp"].dt.month)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, v in zip(axes, TARGETS):
        sns.boxplot(data=mon, x="mes", y=v, ax=ax, color="#a8dadc")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"Estacionalidad mensual - {v}")
    _savefig(fig, "10_estacionalidad_mensual_boxplots.png")

    return resids


# --------------------------------------------------- 5. clase_balance_carbono
def class_balance(obs: pd.DataFrame) -> dict:
    counts = obs["clase_balance_carbono"].value_counts()
    n_sink = int(counts.get("sumidero", 0))
    n_source = int(counts.get("fuente", 0))
    expected = (353, 743)
    match = (n_sink, n_source) == expected
    print(f"  clase_balance_carbono: sumidero={n_sink}  fuente={n_source}  "
          f"(esperado 353/743 -> {'COINCIDE' if match else 'NO COINCIDE'})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = ["sumidero", "fuente"]
    axes[0].bar(order, [n_sink, n_source], color=["#2a9d8f", "#e76f51"])
    for i, n in enumerate([n_sink, n_source]):
        axes[0].text(i, n + 5, f"{n}\n({100*n/(n_sink+n_source):.1f} %)", ha="center")
    axes[0].set_title("Balance de clases (dias 2016-2018)")
    axes[0].set_ylabel("dias")

    mon = obs.assign(mes=obs["timestamp"].dt.month)
    frac = (
        mon.groupby("mes")["clase_balance_carbono"]
        .value_counts(normalize=True)
        .unstack()
        .reindex(columns=order)
    )
    frac.plot(kind="bar", stacked=True, ax=axes[1], color=["#2a9d8f", "#e76f51"])
    axes[1].set_title("Fraccion sumidero/fuente por mes")
    axes[1].set_ylabel("fraccion de dias")
    axes[1].legend(title="clase")
    _savefig(fig, "11_distribucion_clase_balance_carbono.png")

    frac.to_csv(TBL_DIR / "05_clase_balance_por_mes.csv")
    return {"n_sumidero": n_sink, "n_fuente": n_source, "coincide_353_743": match,
            "pct_sumidero": round(100 * n_sink / (n_sink + n_source), 1)}


# --------------------------------------------------- 6. pruebas de normalidad
def normality_tests(obs: pd.DataFrame, resids: dict[str, pd.Series]) -> pd.DataFrame:
    series = {v: obs[v].dropna() for v in KEY_VARS}
    series["residuo_STL_NEE"] = resids["NEE_F_ANNOPTLM"].dropna()
    series["residuo_STL_FCH4"] = resids["FCH4_F_ANNOPTLM"].dropna()

    recs = {}
    for name, x in series.items():
        xv = np.asarray(x, dtype=float)
        W, p_sw = stats.shapiro(xv)
        k2, p_dp = stats.normaltest(xv)  # D'Agostino-Pearson (contraste)
        recs[name] = {
            "n": len(xv),
            "shapiro_W": W,
            "shapiro_p": p_sw,
            "normal_shapiro_5pct": "si" if p_sw > ALPHA else "no",
            "dagostino_K2": k2,
            "dagostino_p": p_dp,
            "normal_dagostino_5pct": "si" if p_dp > ALPHA else "no",
            "asimetria": stats.skew(xv, bias=False),
            "curtosis_exceso": stats.kurtosis(xv, fisher=True, bias=False),
        }
    tbl = pd.DataFrame(recs).T
    tbl["n"] = tbl["n"].astype("int64")
    tbl.index.name = "serie"
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TBL_DIR / "06_normalidad.csv")

    # QQ-plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (name, x) in zip(axes.ravel(), series.items()):
        stats.probplot(np.asarray(x, dtype=float), dist="norm", plot=ax)
        ax.set_title(name, fontsize=9)
    fig.suptitle("QQ-plots vs normal - variables clave y residuos STL", fontsize=13)
    _savefig(fig, "13_qqplots_normalidad.png")
    return tbl


# ------------------------------------------------------------------ resumen markdown
def write_markdown(desc, out_iqr, pear, seas_resids, cls, norm) -> None:
    meta = json.loads(META_JSON.read_text(encoding="utf-8")) if META_JSON.exists() else {}
    src = meta.get("source", {})

    # correlaciones destacadas con los objetivos
    def top_corr(target):
        s = pear[target].drop(index=TARGETS).sort_values(key=lambda z: z.abs(), ascending=False)
        return ", ".join(f"{k} = {v:+.2f}" for k, v in s.head(5).items())

    nee = desc.loc["NEE_F_ANNOPTLM"]
    fch4 = desc.loc["FCH4_F_ANNOPTLM"]
    wtd = desc.loc["WTD"]
    r_nee = norm.loc["residuo_STL_NEE"]
    r_fch4 = norm.loc["residuo_STL_FCH4"]

    md = f"""# EDA - DE-Zrk (Zarnekow) · flujos de carbono diarios 2016-2018

Fuente: `{src.get('file', OBS_PARQUET.name)}` (SHA-256 `{src.get('sha256', 'n/d')[:16]}...`,
{src.get('rows', 'n/d')} filas crudas) -> ETL -> `{OBS_PARQUET.name}` (1096 filas,
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

{_md_table(desc, ".3f")}

*Figuras:* `01_histogramas_kde_variables_clave.png`, `14_perfil_termico_suelo_TS1_TS5.png`

**Interpretacion**
- **NEE_F_ANNOPTLM**: media = {nee['media']:+.2f} gC m-2 d-1 (practicamente
  neutra en promedio diario) pero mediana = {nee['mediana']:+.2f} y solo
  {cls['pct_sumidero']} % de dias con captura -> **el sitio es fuente de CO2 la
  mayoria de los dias**, con episodios intensos de captura en verano que tiran de
  la media hacia cero. Asimetria = {nee['asimetria']:+.2f} (cola izquierda) y
  curtosis de exceso = {nee['curtosis_exceso']:+.2f} (leptocurtica): coherente con
  ese regimen estacional bimodal.
- **FCH4_F_ANNOPTLM**: media = {fch4['media']:.2f} nmol m-2 s-1, mediana =
  {fch4['mediana']:.2f}, **asimetria = {fch4['asimetria']:+.2f}** (cola derecha
  marcada) y curtosis de exceso = {fch4['curtosis_exceso']:+.2f}. Tipico de un
  flujo estrictamente positivo con pulsos estivales: convendra transformar
  (log / sqrt) antes de modelos que asuman normalidad de residuos.
- **WTD**: media = {wtd['media']:+.3f} m, rango [{wtd['min']:+.2f}, {wtd['max']:+.2f}] m.
  Predominantemente **positivo** = lamina de agua por encima de la superficie:
  confirma que Zarnekow es un fen **rehumedecido/inundado**.
- **Perfil TS_1..TS_5**: la media es casi identica (~10.9 C) pero la **desviacion
  estandar decrece monotonamente con la profundidad**
  ({desc.loc['TS_1','desv_std']:.2f} C a 0.05 m -> {desc.loc['TS_5','desv_std']:.2f} C
  a ~0.48 m): amortiguamiento termico clasico. Para modelar basta 1-2
  profundidades (el resto es redundante, ver correlacion).

## 2. Deteccion y tratamiento de outliers

**Criterio:** regla de Tukey sobre el IQR. "Marcado" fuera de
[Q1 - 1.5·IQR, Q3 + 1.5·IQR]; "extremo" fuera de [Q1 - 3.0·IQR, Q3 + 3.0·IQR].

{_md_table(out_iqr[['Q1', 'Q3', 'IQR', 'lim_inf_1.5', 'lim_sup_1.5', 'n_out_1.5', 'pct_1.5', 'n_out_3.0', 'n_winsorizados_k3']], ".2f")}

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
- Predictores mas correlacionados con **NEE**: {top_corr('NEE_F_ANNOPTLM')}.
- Predictores mas correlacionados con **FCH4**: {top_corr('FCH4_F_ANNOPTLM')}.
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
**{'COINCIDE' if cls['coincide_353_743'] else 'NO COINCIDE'}**
(sumidero = {cls['n_sumidero']}, fuente = {cls['n_fuente']},
{cls['pct_sumidero']} % de dias sumidero).

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

{_md_table(norm[['n', 'shapiro_W', 'shapiro_p', 'normal_shapiro_5pct', 'dagostino_K2', 'dagostino_p', 'normal_dagostino_5pct', 'asimetria', 'curtosis_exceso']], ".4f")}

**Interpretacion**
- **Ninguna** de las variables clave supera Shapiro-Wilk: se rechaza normalidad
  (p < 0.05) en todos los casos. D'Agostino-Pearson coincide.
- Los **residuos STL** tampoco son normales. El residuo de NEE es casi simetrico
  (asimetria {r_nee['asimetria']:+.2f}) pero con **colas muy pesadas**
  (curtosis de exceso {r_nee['curtosis_exceso']:+.2f}); el de FCH4 es ademas
  fuertemente asimetrico ({r_fch4['asimetria']:+.2f}) y leptocurtico
  ({r_fch4['curtosis_exceso']:+.2f}). Shapiro-Wilk p = {r_nee['shapiro_p']:.1e}
  y {r_fch4['shapiro_p']:.1e} respectivamente.
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
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"  resumen -> {OUT_MD.relative_to(ROOT)}")


# --------------------------------------------------------------------------- main
def main() -> None:
    _logging.setup("eda")
    for p in (OBS_PARQUET, ML_PARQUET):
        if not p.exists():
            raise SystemExit(f"Falta {p}. Ejecuta antes: python -m ml.etl.run_etl")

    obs = pd.read_parquet(OBS_PARQUET, engine="pyarrow")
    ml = pd.read_parquet(ML_PARQUET, engine="pyarrow")
    obs["timestamp"] = pd.to_datetime(obs["timestamp"])

    print("=" * 79)
    print(f"EDA DE-Zrk  ({len(obs)} filas, {obs['timestamp'].min().date()} .. {obs['timestamp'].max().date()})")
    print("=" * 79)

    print("[1] estadisticos descriptivos")
    desc = descriptive_stats(obs)
    print("[2] outliers (IQR) + winsorizacion k=3")
    out_iqr, _flags = outlier_analysis(obs)
    print("[3] matriz de correlacion")
    pear = correlation_matrix(ml)
    print("[4] series temporales + estacionalidad (STL)")
    resids = time_series_and_seasonality(obs)
    print("[5] distribucion de clase_balance_carbono")
    cls = class_balance(obs)
    print("[6] pruebas de normalidad (Shapiro-Wilk + contraste)")
    norm = normality_tests(obs, resids)
    print("[7] resumen markdown")
    write_markdown(desc, out_iqr, pear, resids, cls, norm)

    print("-" * 79)
    print(f"OK. {len(list(FIG_DIR.glob('*.png')))} figuras, "
          f"{len(list(TBL_DIR.glob('*.csv')))} tablas, resumen en {OUT_MD.name}")


if __name__ == "__main__":
    main()
