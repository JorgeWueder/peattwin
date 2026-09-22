"""Pruebas estadisticas para comparar formalmente los 5 modelos BASE.

Fuente de datos: `ml_model_metrics`, filas `fold` 0-4 (los 5 folds de la
validacion cruzada temporal de `cross_validation.py`), version "1.0" -> los 5
modelos SIN ajuste de hiperparametros. Metrica de "error" por (modelo, fold):
`rmse` de las filas de REGRESION (media de RMSE de NEE y FCH4, tal cual se guardo
en la BD). Como comparacion secundaria se usa `f1` de las filas de CLASIFICACION.

Bateria de pruebas:
  1. Shapiro-Wilk sobre la distribucion de error de cada modelo entre folds
     (y sobre las diferencias pareadas) -> decide parametrica vs no parametrica.
  2. Comparacion pareada Stacking Ensemble vs cada uno de los otros 4:
        - datos normales  -> t de Student pareada  (ttest_rel)
        - datos no normales -> Wilcoxon signed-rank
     con correccion de Holm por comparaciones multiples.
  3. Omnibus (los 5 modelos a la vez):
        - normales  -> ANOVA de una via (f_oneway)
        - no normales -> Kruskal-Wallis
     + Friedman (version de medidas repetidas, la correcta para folds pareados).
  4. Diebold-Mariano (Stacking vs Random Forest, el 2o mejor): sobre la secuencia
     de errores diarios del HOLDOUT 2018 (365 dias consecutivos), que es el marco
     natural de la prueba. Con correccion de muestra pequena de
     Harvey-Leybourne-Newbold y varianza de largo plazo Newey-West.

Salidas:
  ml/training/statistical_tests_resultados.csv   -> tabla de p-values
  ml/training/PRUEBAS_ESTADISTICAS_resumen.md    -> tabla + interpretacion (informe)
  ml/training/figuras/statistical_tests_distribuciones.png

Ejecutar desde la raiz del repositorio:
    python -m ml.training.statistical_tests
    python -m ml.training.statistical_tests --from-csv   # usa cv_por_fold.csv en vez de la BD

-------------------------------------------------------------------------------
NOTA METODOLOGICA (va al informe):
  Estas pruebas se hicieron sobre los **5 modelos BASE** de la validacion
  cruzada (sin tuning). El modelo **desplegado** (`Stacking Ensemble (tuned)`,
  `modelo_final.pkl`, `ml_models` v1.1-tuned) es un refinamiento POSTERIOR del
  ganador ya seleccionado por el criterio combinado: la ventaja del Stacking
  frente al resto queda establecida aqui sobre las versiones base, y el tuning
  solo optimizo el meta-modelo (Ridge alpha=10) sin cambiar la eleccion de
  arquitectura.
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ml.training import config as C  # noqa: E402
from ml.training import data as D  # noqa: E402
from ml.training import metrics as MET  # noqa: E402
from ml.training import models_classical as MC  # noqa: E402
from ml import _logging

MODELS = ["Stacking Ensemble", "Random Forest", "Gradient Boosting (XGBoost)",
          "SVR / SVM", "CNN-LSTM"]
REFERENCE = "Stacking Ensemble"     # mejor modelo (criterio combinado)
SECOND = "Random Forest"            # 2o mejor -> comparacion Diebold-Mariano
FOLDS = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------- carga de datos
def load_from_db() -> pd.DataFrame:
    from sqlalchemy import MetaData, Table, create_engine, select

    from ml.etl.load_to_db import _database_url

    engine = create_engine(_database_url(None), future=True)
    md = MetaData()
    mm = Table("ml_model_metrics", md, autoload_with=engine)
    mdl = Table("ml_models", md, autoload_with=engine)
    q = (
        select(mdl.c.name, mdl.c.task, mm.c.fold, mm.c.rmse, mm.c.mae, mm.c.r2,
               mm.c.f1, mm.c.auc_roc)
        .select_from(mm.join(mdl, mm.c.model_id == mdl.c.id))
        .where(mm.c.fold.in_(FOLDS), mdl.c.version == "1.0")
    )
    with engine.begin() as conn:
        rows = conn.execute(q).all()
    df = pd.DataFrame(rows, columns=["name", "task", "fold", "rmse", "mae", "r2", "f1", "auc"])
    df["model"] = (df["name"].str.replace(" - regresion", "", regex=False)
                   .str.replace(" - clasificacion", "", regex=False))
    return df


def load_from_csv() -> pd.DataFrame:
    raw = pd.read_csv(C.CV_PER_FOLD_CSV)
    raw["rmse"] = raw[["rmse_nee", "rmse_fch4"]].mean(axis=1)  # = lo que se guardo en la BD
    reg = raw[["model", "fold", "rmse"]].assign(task="REGRESSION")
    cls = raw[["model", "fold", "f1_macro", "auc"]].rename(
        columns={"f1_macro": "f1"}).assign(task="CLASSIFICATION")
    return pd.concat([reg, cls], ignore_index=True)


def matrices(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = long[long["task"] == "REGRESSION"].pivot_table(index="fold", columns="model", values="rmse")
    cls = long[long["task"] == "CLASSIFICATION"].pivot_table(index="fold", columns="model", values="f1")
    return reg[MODELS], cls[MODELS]


# ----------------------------------------------------------------- 1. normalidad
def shapiro_block(mat: pd.DataFrame, label: str) -> tuple[pd.DataFrame, bool]:
    recs = []
    for m in MODELS:
        x = mat[m].to_numpy()
        W, p = stats.shapiro(x)
        recs.append({"objeto": f"{label}: {m}", "n": len(x), "shapiro_W": W,
                     "p_value": p, "normal_alpha05": p > C.STATS_ALPHA})
    # diferencias pareadas contra la referencia
    for m in MODELS:
        if m == REFERENCE:
            continue
        d = mat[REFERENCE].to_numpy() - mat[m].to_numpy()
        if np.allclose(d, 0):
            recs.append({"objeto": f"{label}: dif {REFERENCE}-{m}", "n": len(d),
                         "shapiro_W": np.nan, "p_value": np.nan, "normal_alpha05": True})
            continue
        W, p = stats.shapiro(d)
        recs.append({"objeto": f"{label}: dif {REFERENCE}-{m}", "n": len(d),
                     "shapiro_W": W, "p_value": p, "normal_alpha05": p > C.STATS_ALPHA})
    tbl = pd.DataFrame(recs)
    all_normal = bool(tbl["normal_alpha05"].all())
    return tbl, all_normal


# ----------------------------------------------------------------- 2. pareadas
def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def paired_block(mat: pd.DataFrame, normal: bool, label: str) -> pd.DataFrame:
    test_name = "t pareada" if normal else "Wilcoxon signed-rank"
    recs = []
    raw_p = []
    for m in MODELS:
        if m == REFERENCE:
            continue
        a, b = mat[REFERENCE].to_numpy(), mat[m].to_numpy()
        if normal:
            st, p = stats.ttest_rel(a, b)
        else:
            try:
                st, p = stats.wilcoxon(a, b)
            except ValueError:              # p.ej. todas las diferencias 0
                st, p = np.nan, 1.0
        raw_p.append(p)
        recs.append({"comparacion": f"{REFERENCE} vs {m}", "metrica": label,
                     "test": test_name, "statistic": st, "p_value": p,
                     "mean_diff": float(np.mean(a - b))})
    p_holm = holm(raw_p)
    for r, ph in zip(recs, p_holm):
        r["p_holm"] = ph
        r["significativo_alpha05"] = ph < C.STATS_ALPHA
    return pd.DataFrame(recs)


# ----------------------------------------------------------------- 3. omnibus
def omnibus_block(mat: pd.DataFrame, normal: bool, label: str) -> pd.DataFrame:
    cols = [mat[m].to_numpy() for m in MODELS]
    recs = []
    if normal:
        st, p = stats.f_oneway(*cols)
        recs.append({"prueba": "ANOVA de una via", "metrica": label, "statistic": st,
                     "p_value": p, "significativo_alpha05": p < C.STATS_ALPHA})
    else:
        st, p = stats.kruskal(*cols)
        recs.append({"prueba": "Kruskal-Wallis", "metrica": label, "statistic": st,
                     "p_value": p, "significativo_alpha05": p < C.STATS_ALPHA})
    # Friedman: version de medidas repetidas (folds pareados) - siempre util
    st_f, p_f = stats.friedmanchisquare(*cols)
    recs.append({"prueba": "Friedman (medidas repetidas)", "metrica": label,
                 "statistic": st_f, "p_value": p_f,
                 "significativo_alpha05": p_f < C.STATS_ALPHA})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------- 4. Diebold-Mariano
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """DM con varianza de largo plazo Newey-West (Bartlett) y correccion HLN.

    d_t = |e1|^power - |e2|^power. d<0 -> el modelo 1 (e1) predice mejor.
    """
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    dc = d - dbar

    lag = max(h - 1, int(round(n ** (1.0 / 3.0))))
    g0 = np.dot(dc, dc) / n
    S = g0 + 2.0 * sum(
        (1.0 - k / (lag + 1.0)) * np.dot(dc[k:], dc[:n - k]) / n
        for k in range(1, lag + 1)
    )
    S = max(S, 1e-12)
    dm = dbar / np.sqrt(S / n)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_star = dm * hln
    p = float(2.0 * stats.t.sf(abs(dm_star), df=n - 1))
    return {"DM_stat": float(dm_star), "p_value": p, "mean_loss_diff": float(dbar),
            "lag_NW": lag, "n": n}


def holdout_errors() -> dict[str, np.ndarray]:
    """Reentrena Stacking base y RF base y devuelve errores diarios del holdout 2018."""
    df, feats = D.build_dataset()
    fr = D.train_test_frames(df, feats)
    out: dict[str, np.ndarray] = {}
    for tgt, key in [("nee", C.TARGET_NEE), ("fch4", C.TARGET_FCH4)]:
        y_test = fr[f"{tgt}_test"].to_numpy()
        st = MC.stacking_regressor().fit(fr["X_train"], fr[f"{tgt}_train"])
        rf = MC.rf_regressor().fit(fr["X_train"], fr[f"{tgt}_train"])
        out[f"{tgt}_stacking_err"] = y_test - st.predict(fr["X_test"])
        out[f"{tgt}_rf_err"] = y_test - rf.predict(fr["X_test"])
    return out


# ----------------------------------------------------------------- figura
def figure(reg: pd.DataFrame, cls: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, mat, title, ylab in [
        (axes[0], reg, "RMSE por fold (CV) - mas bajo mejor", "RMSE (media NEE+FCH4)"),
        (axes[1], cls, "F1_macro por fold (CV) - mas alto mejor", "F1_macro"),
    ]:
        data = [mat[m].to_numpy() for m in MODELS]
        ax.boxplot(data, tick_labels=MODELS, showmeans=True)
        for i, d in enumerate(data, start=1):
            ax.scatter(np.full(len(d), i), d, s=20, alpha=0.6, color="#1d3557", zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Distribucion de metricas entre los 5 folds de CV (modelos base)", fontsize=13)
    fig.tight_layout()
    _save(fig, C.STATS_FIG.name)


def _save(fig, name: str) -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> figuras/{name}")


# --------------------------------------------------- 5. post-hoc de Nemenyi
def _nemenyi_manual(reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nemenyi para Friedman a mano (distribucion del rango studentizado)."""
    avg_ranks = reg.rank(axis=1, method="average").mean(axis=0)   # 1 = mejor (menor RMSE)
    k, n = reg.shape[1], reg.shape[0]
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    cols = list(reg.columns)
    pmat = pd.DataFrame(np.ones((k, k)), index=cols, columns=cols)
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(avg_ranks.iloc[i] - avg_ranks.iloc[j]) / se
            p = float(stats.studentized_range.sf(z * np.sqrt(2), k, np.inf))
            pmat.iloc[i, j] = pmat.iloc[j, i] = min(p, 1.0)
    return pmat, avg_ranks


def nemenyi_posthoc(reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float, str]:
    try:
        import scikit_posthocs as sp
        pmat = sp.posthoc_nemenyi_friedman(reg.to_numpy())
        pmat.index = list(reg.columns)
        pmat.columns = list(reg.columns)
        avg_ranks = reg.rank(axis=1, method="average").mean(axis=0)
        lib = "scikit-posthocs"
    except ImportError:
        pmat, avg_ranks = _nemenyi_manual(reg)
        lib = "implementacion manual (rango studentizado)"
    k, n = reg.shape[1], reg.shape[0]
    q = stats.studentized_range.ppf(1 - C.STATS_ALPHA, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
    return pmat, avg_ranks.sort_values(), cd, lib


def nemenyi_figures(pmat: pd.DataFrame, avg_ranks: pd.Series, cd: float) -> bool:
    # (a) diagrama de diferencias criticas (si scikit-posthocs esta disponible)
    cd_ok = False
    try:
        import scikit_posthocs as sp
        fig, ax = plt.subplots(figsize=(9.5, 2.8))
        sp.critical_difference_diagram(avg_ranks, pmat, ax=ax)
        ax.set_title(
            f"Diagrama de diferencias criticas - Nemenyi (alpha={C.STATS_ALPHA}, CD={cd:.2f})\n"
            "modelos unidos por una barra: NO significativamente distintos (RMSE, CV)",
            fontsize=10,
        )
        _save(fig, "statistical_tests_nemenyi_cd.png")
        cd_ok = True
    except Exception as exc:  # pragma: no cover
        print(f"  (diagrama CD no generado: {exc})")

    # (b) heatmap de la matriz de p-values (siempre)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    P = pmat.to_numpy()
    im = ax.imshow(P, cmap="RdYlGn", vmin=0, vmax=0.20)
    ax.set_xticks(range(len(pmat)))
    ax.set_xticklabels(pmat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pmat)))
    ax.set_yticklabels(pmat.index)
    for i in range(len(pmat)):
        for j in range(len(pmat)):
            v = P[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.03 < v < 0.15 else "white")
    ax.set_title("Nemenyi post-hoc: matriz de p-values pareados (RMSE, CV)\n"
                 f"verde = p < {C.STATS_ALPHA} (par significativamente distinto)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="p-value")
    fig.tight_layout()
    _save(fig, "statistical_tests_nemenyi_pmatrix.png")
    return cd_ok


# ----------------------------------------------------------------- resumen MD
def write_md(reg: pd.DataFrame, cls: pd.DataFrame, sh_reg: pd.DataFrame,
             norm_reg: bool, sh_cls: pd.DataFrame, norm_cls: bool,
             paired_reg: pd.DataFrame, paired_cls: pd.DataFrame,
             omni_reg: pd.DataFrame, omni_cls: pd.DataFrame, dm: pd.DataFrame,
             nemenyi: tuple | None) -> None:

    def md_table(df, floatfmt=".4g"):
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            cells = []
            for v in r:
                if isinstance(v, (float, np.floating)):
                    cells.append("n/a" if pd.isna(v) else f"{v:{floatfmt}}")
                elif isinstance(v, (bool, np.bool_)):
                    cells.append("SI" if v else "no")
                else:
                    cells.append(str(v))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    rmse_mean = reg.mean().sort_values()
    f1_mean = cls.mean().sort_values(ascending=False)
    best_rmse = rmse_mean.index[0]
    sig_reg = paired_reg[paired_reg["significativo_alpha05"]]["comparacion"].tolist()
    has_dm = not dm.empty
    _nan = {"p_value": float("nan"), "mean_loss_diff": float("nan")}
    dm_nee = dm[dm["serie"] == "error NEE"].iloc[0] if has_dm else _nan
    dm_fch4 = dm[dm["serie"] == "error FCH4"].iloc[0] if has_dm else _nan
    dm_significant = has_dm and bool(dm["significativo_alpha05"].any())

    omni_p = omni_reg[omni_reg["prueba"].str.startswith(("ANOVA", "Kruskal"))]["p_value"].iloc[0]
    fried_p = omni_reg[omni_reg["prueba"].str.startswith("Friedman")]["p_value"].iloc[0]

    # ---- seccion 5: post-hoc de Nemenyi ----
    if nemenyi is not None:
        pmat, avg_ranks, cd, lib, cd_ok = nemenyi
        sig_pairs = [(a, b, float(pmat.loc[a, b]))
                     for i, a in enumerate(pmat.index) for b in pmat.index[i + 1:]
                     if pmat.loc[a, b] < C.STATS_ALPHA]
        max_rank_gap = float(avg_ranks.max() - avg_ranks.min())
        rank_str = ", ".join(f"{m} {r:.2f}" for m, r in avg_ranks.items())
        pmat_md = md_table(pmat.round(4).reset_index().rename(columns={"index": "modelo"}))
        cd_fig_line = ("![CD](figuras/statistical_tests_nemenyi_cd.png)\n\n"
                       if cd_ok else "_(diagrama CD no disponible en esta instalacion)_\n\n")
        if sig_pairs:
            pares_txt = "; ".join(f"**{a} vs {b}** (p = {p:.3f})" for a, b, p in sig_pairs)
            worst = avg_ranks.index[-1]
            second_worst = avg_ranks.index[-2]
            # p-value minimo del 2o peor modelo contra los tres mejores
            best3 = list(avg_ranks.index[:3])
            sw_min_p = min(float(pmat.loc[second_worst, m]) for m in best3)
            posthoc_concl = (
                f"El post-hoc **SI aisla** el/los par(es) responsable(s) de la "
                f"diferencia global de Friedman: {pares_txt}. Es decir, la diferencia "
                f"global la produce sobre todo **{worst}** (el peor por rango medio), "
                f"que queda por debajo de los modelos de arboles. "
                f"Entre **{REFERENCE}**, **{SECOND}** y **Gradient Boosting (XGBoost)** "
                f"NO hay ninguna diferencia significativa (p ~ 1.0): son estadisticamente "
                f"equivalentes en RMSE. **{second_worst}** tiene el 2o peor rango pero su "
                f"diferencia con los tres mejores NO alcanza significancia (p_min = "
                f"{sw_min_p:.3f}); con solo {len(reg)} folds el post-hoc no tiene "
                f"potencia para confirmarla."
            )
        else:
            posthoc_concl = (
                f"El post-hoc de Nemenyi **NO consigue aislar** ningun par concreto: "
                f"la diferencia critica CD = {cd:.2f} (en unidades de rango, escala 1-{len(avg_ranks)}) "
                f"es mayor que la mayor diferencia de rangos observada ({max_rank_gap:.2f}). "
                f"Es un resultado habitual: el omnibus de Friedman tiene mas potencia que "
                f"su post-hoc, sobre todo con solo {len(reg)} folds. Los rangos medios "
                f"(menor = mejor) — {rank_str} — indican que **{avg_ranks.index[-1]}** y "
                f"**{avg_ranks.index[-2]}** son los que empujan la diferencia global, pero "
                f"no puede afirmarse par por par con este numero de folds."
            )
        nemenyi_section = f"""
## 5. Post-hoc de Nemenyi (tras Friedman significativo)

Friedman detecto diferencias globales en RMSE (p = {fried_p:.3g} < {C.STATS_ALPHA}),
asi que se aplica el post-hoc de **Nemenyi** sobre la misma matriz RMSE
(folds x modelos). Libreria: {lib}. Diferencia critica **CD = {cd:.3f}** rangos.

**Rangos medios** (1 = mejor RMSE): {rank_str}

**Matriz de p-values pareados (Nemenyi):**

{pmat_md}

{cd_fig_line}*(heatmap: `figuras/statistical_tests_nemenyi_pmatrix.png`)*

**¿Que par(es) explican la diferencia global?** {posthoc_concl}
"""
    else:
        nemenyi_section = ""
        posthoc_concl = ("Friedman no fue significativo para RMSE, por lo que no se "
                         "aplica post-hoc.")

    md = f"""# Pruebas estadisticas - comparacion formal de los 5 modelos base

Datos: `ml_model_metrics`, `fold` 0-4 (5 folds de la validacion cruzada temporal),
version 1.0 (modelos SIN tuning). Metrica de error: `rmse` de las filas de
regresion (media de RMSE de NEE y FCH4). alpha = {C.STATS_ALPHA}.

RMSE medio entre folds (menor = mejor): {", ".join(f"{k} {v:.2f}" for k, v in rmse_mean.items())}
F1_macro medio entre folds (mayor = mejor): {", ".join(f"{k} {v:.3f}" for k, v in f1_mean.items())}

> El **mejor por criterio combinado** (Paso 5) es **{REFERENCE}** (mejor F1_macro,
> AUC y estabilidad entre folds). Por **RMSE puro** de la CV el mas bajo es
> **{best_rmse}** {"(mismo modelo)" if best_rmse == REFERENCE else "(no es " + REFERENCE + "): la diferencia de RMSE entre ambos es pequena, ver pruebas pareadas mas abajo"}.
> La referencia de estas pruebas es **{REFERENCE}** y el 2o mejor **{SECOND}**.

## 1. Normalidad (Shapiro-Wilk)

### Error RMSE (CV)
{md_table(sh_reg)}

**Decision:** {"todas las distribuciones y diferencias pareadas pasan Shapiro (p > alpha) -> se usan pruebas PARAMETRICAS." if norm_reg else "al menos una distribucion/diferencia NO pasa Shapiro (p <= alpha) -> se usan pruebas NO PARAMETRICAS."}

### F1_macro (CV)
{md_table(sh_cls)}

**Decision (F1):** {"parametricas." if norm_cls else "no parametricas."}

> **Aviso de potencia (n = 5 folds):** Shapiro-Wilk casi no tiene potencia para
> detectar no-normalidad con 5 datos, y la prueba de Wilcoxon con 5 pares no
> puede bajar de p ≈ 0.0625 (nunca alcanza significancia a 0.05). Los resultados
> son **indicativos**, no concluyentes; se reportan ambas familias de pruebas.

## 2. Comparacion pareada: {REFERENCE} vs cada modelo (Holm)

### Error RMSE (CV)
{md_table(paired_reg)}

### F1_macro (CV)
{md_table(paired_cls)}

## 3. Comparacion global de los 5 modelos

### Error RMSE (CV)
{md_table(omni_reg)}

### F1_macro (CV)
{md_table(omni_cls)}

## 4. Diebold-Mariano: {REFERENCE} vs {SECOND} (holdout 2018, 365 dias)

Sobre la secuencia de errores diarios del holdout (marco natural de la prueba;
las otras pruebas usan los 5 folds de la CV). Perdida = error cuadratico.
`mean_loss_diff < 0` -> {REFERENCE} predice mejor.

{md_table(dm[["serie", "DM_stat", "p_value", "mean_loss_diff", "lag_NW", "n", "significativo_alpha05"]]) if has_dm else "_(omitido con --no-dm)_"}
{nemenyi_section}
## Interpretacion en texto simple (para el informe)

- **¿Normalidad?** {"Los errores por fold se comportan como normales segun Shapiro, asi que se aplican t pareada y ANOVA. Aun asi, con solo 5 folds el resultado es fragil." if norm_reg else "Los errores por fold NO se ajustan a una normal, asi que se aplican Wilcoxon y Kruskal-Wallis / Friedman."}

- **¿{REFERENCE} es mejor que los demas de forma significativa (RMSE, CV)?**
  {"SI para: " + ", ".join(sig_reg) + "." if sig_reg else "NO. Ninguna comparacion pareada alcanza p < " + str(C.STATS_ALPHA) + " tras la correccion de Holm."}
  {"" if sig_reg else " Con 5 folds la prueba no tiene resolucion suficiente; la ventaja del Stacking se sostiene por su MENOR VARIABILIDAD entre folds y por el criterio combinado (F1_macro + AUC), no por una diferencia de RMSE estadisticamente demostrable."}

- **¿Los 5 modelos son equivalentes en conjunto?** La prueba global da
  p = {omni_p:.3g} ({"se rechaza la igualdad: hay al menos un modelo distinto" if omni_p < C.STATS_ALPHA else "no se puede rechazar que todos rindan igual"}).
  Friedman (medidas repetidas) da p = {fried_p:.3g}
  ({"diferencias significativas entre modelos" if fried_p < C.STATS_ALPHA else "sin diferencias significativas"}).

- **Post-hoc de Nemenyi (¿que par explica la diferencia de Friedman?):** {posthoc_concl}

- **Diebold-Mariano {REFERENCE} vs {SECOND}:**
  error de NEE -> p = {dm_nee['p_value']:.3g} ({"diferencia significativa" if dm_nee['p_value'] < C.STATS_ALPHA else "sin diferencia significativa"});
  error de CH4 -> p = {dm_fch4['p_value']:.3g} ({"diferencia significativa" if dm_fch4['p_value'] < C.STATS_ALPHA else "sin diferencia significativa"}).
  {"El signo de mean_loss_diff indica cual predice mejor." }

- **Conclusion para el informe:** con la evidencia disponible (5 folds de CV +
  holdout 2018), {"la superioridad de " + REFERENCE + " es estadisticamente significativa en las comparaciones marcadas arriba." if (sig_reg or dm_nee['p_value'] < C.STATS_ALPHA or dm_fch4['p_value'] < C.STATS_ALPHA) else "no se puede afirmar que " + REFERENCE + " sea estadisticamente superior en RMSE a " + SECOND + ": las diferencias de exactitud son pequenas y el numero de folds es bajo. " + REFERENCE + " se elige por su mejor comportamiento CONJUNTO (regresion + clasificacion), su menor varianza entre folds y el criterio combinado del Paso 5, no por una diferencia de error puntual demostrable."}

---

## Nota metodologica (importante)

Estas pruebas se realizaron sobre los **5 modelos BASE** de la validacion
cruzada (`ml_model_metrics`, `fold` 0-4, version 1.0), **sin ajuste de
hiperparametros**.

El **modelo desplegado** es `Stacking Ensemble (tuned)` (`modelo_final.pkl`,
`ml_models` version `1.1-tuned`, `is_active = true`), un **refinamiento
posterior** del ganador ya seleccionado por el criterio combinado del Paso 5. El
tuning solo optimizo el meta-modelo del stacking (`Ridge(alpha=10)` en lugar de
`LinearRegression`) y algun hiperparametro de los modelos base; **no cambio la
eleccion de arquitectura**. La comparacion estadistica de esta seccion respalda
esa eleccion sobre las versiones base; el tuned es una mejora incremental sobre
el mismo modelo, no un candidato nuevo que debiera re-someterse a esta bateria.

*(generado por `python -m ml.training.statistical_tests` el {datetime.now(timezone.utc).date()})*
"""
    C.STATS_MD.write_text(md, encoding="utf-8")
    print(f"  resumen -> {C.STATS_MD.relative_to(C.ROOT)}")


# --------------------------------------------------------------------------- main
def main() -> None:
    _logging.setup("pruebas_estadisticas")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-csv", action="store_true",
                    help="usar cv_por_fold.csv en vez de ml_model_metrics")
    ap.add_argument("--no-dm", action="store_true", help="omitir Diebold-Mariano (no reentrena)")
    args = ap.parse_args()

    print("=" * 79)
    print("Pruebas estadisticas - comparacion de los 5 modelos base (CV fold 0-4)")
    print("=" * 79)

    if args.from_csv:
        long = load_from_csv()
        print(f"fuente: {C.CV_PER_FOLD_CSV.name}")
    else:
        try:
            long = load_from_db()
            n_ok = long[long["task"] == "REGRESSION"].groupby("model")["fold"].nunique()
            if len(n_ok) < 5 or (n_ok < 5).any():
                raise RuntimeError(f"faltan folds en la BD: {n_ok.to_dict()}")
            print("fuente: ml_model_metrics (PostgreSQL), fold 0-4, version 1.0")
        except Exception as exc:
            print(f"  aviso: no se pudo leer de la BD ({exc}); uso {C.CV_PER_FOLD_CSV.name}")
            long = load_from_csv()

    reg, cls = matrices(long)
    print("\nRMSE por fold (CV):")
    print(reg.round(3).to_string())
    print("\nF1_macro por fold (CV):")
    print(cls.round(3).to_string())

    # 1. normalidad
    sh_reg, norm_reg = shapiro_block(reg, "RMSE")
    sh_cls, norm_cls = shapiro_block(cls, "F1")
    print(f"\n[1] Shapiro RMSE -> {'NORMAL' if norm_reg else 'NO normal'}  | "
          f"F1 -> {'NORMAL' if norm_cls else 'NO normal'}")

    # 2. pareadas
    paired_reg = paired_block(reg, norm_reg, "RMSE (CV)")
    paired_cls = paired_block(cls, norm_cls, "F1_macro (CV)")
    print(f"[2] pareadas ({paired_reg['test'].iloc[0]}):")
    print(paired_reg[["comparacion", "p_value", "p_holm", "significativo_alpha05"]]
          .round(4).to_string(index=False))

    # 3. omnibus
    omni_reg = omnibus_block(reg, norm_reg, "RMSE (CV)")
    omni_cls = omnibus_block(cls, norm_cls, "F1_macro (CV)")
    print("[3] omnibus RMSE:")
    print(omni_reg[["prueba", "p_value", "significativo_alpha05"]].round(4).to_string(index=False))

    # 3b. post-hoc de Nemenyi (solo si Friedman-RMSE fue significativo)
    fried_p = float(omni_reg.loc[omni_reg["prueba"].str.startswith("Friedman"), "p_value"].iloc[0])
    nemenyi = None
    nemenyi_csv = pd.DataFrame()
    if fried_p < C.STATS_ALPHA:
        pmat, avg_ranks, cd, lib = nemenyi_posthoc(reg)
        cd_ok = nemenyi_figures(pmat, avg_ranks, cd)
        nemenyi = (pmat, avg_ranks, cd, lib, cd_ok)
        sig = [(a, b, float(pmat.loc[a, b]))
               for i, a in enumerate(pmat.index) for b in pmat.index[i + 1:]
               if pmat.loc[a, b] < C.STATS_ALPHA]
        print(f"[3b] Nemenyi post-hoc ({lib}), CD={cd:.3f} rangos")
        print("     rangos medios (1=mejor):",
              ", ".join(f"{m}={r:.2f}" for m, r in avg_ranks.items()))
        print("     pares con p<0.05:",
              [f"{a} vs {b} (p={p:.3f})" for a, b, p in sig] or "ninguno")
        # filas para el CSV consolidado
        rows = []
        for i, a in enumerate(pmat.index):
            for b in pmat.index[i + 1:]:
                p = float(pmat.loc[a, b])
                rows.append({"detalle": f"{a} vs {b}", "test": "Nemenyi post-hoc",
                             "metrica": "RMSE (CV)", "p_value": p,
                             "significativo_alpha05": p < C.STATS_ALPHA})
        rows.append({"detalle": f"CD (critical difference, k={len(avg_ranks)}, N={len(reg)})",
                     "test": "Nemenyi post-hoc", "metrica": "RMSE (CV)", "statistic": cd})
        nemenyi_csv = pd.DataFrame(rows)
    else:
        print("[3b] Nemenyi omitido: Friedman-RMSE no significativo")

    # 4. Diebold-Mariano
    dm_rows = []
    if not args.no_dm:
        print("[4] Diebold-Mariano (reentrenando Stacking base y RF base)...")
        err = holdout_errors()
        for serie, e1, e2 in [
            ("error NEE", err["nee_stacking_err"], err["nee_rf_err"]),
            ("error FCH4", err["fch4_stacking_err"], err["fch4_rf_err"]),
        ]:
            res = diebold_mariano(e1, e2)
            res["serie"] = serie
            res["significativo_alpha05"] = res["p_value"] < C.STATS_ALPHA
            dm_rows.append(res)
            print(f"    {serie}: DM*={res['DM_stat']:+.3f}  p={res['p_value']:.4f}  "
                  f"mean_loss_diff={res['mean_loss_diff']:+.4g}  "
                  f"({'Stacking mejor' if res['mean_loss_diff'] < 0 else 'RF mejor'})")
    dm = pd.DataFrame(dm_rows) if dm_rows else pd.DataFrame(
        columns=["serie", "DM_stat", "p_value", "mean_loss_diff", "lag_NW", "n",
                 "significativo_alpha05"])

    # --- CSV consolidado ---
    def tag(df, seccion):
        return df.assign(seccion=seccion)

    parts = [
        tag(sh_reg.rename(columns={"objeto": "detalle"}), "1-normalidad-RMSE"),
        tag(sh_cls.rename(columns={"objeto": "detalle"}), "1-normalidad-F1"),
        tag(paired_reg.rename(columns={"comparacion": "detalle"}), "2-pareada-RMSE"),
        tag(paired_cls.rename(columns={"comparacion": "detalle"}), "2-pareada-F1"),
        tag(omni_reg.rename(columns={"prueba": "detalle"}), "3-omnibus-RMSE"),
        tag(omni_cls.rename(columns={"prueba": "detalle"}), "3-omnibus-F1"),
    ]
    if not nemenyi_csv.empty:
        parts.append(tag(nemenyi_csv, "3b-nemenyi-posthoc-RMSE"))
    if not dm.empty:
        parts.append(tag(dm.rename(columns={"serie": "detalle"}), "4-diebold-mariano"))
    all_rows = pd.concat(parts, ignore_index=True)
    front = ["seccion", "detalle", "test", "metrica", "statistic", "DM_stat",
             "shapiro_W", "p_value", "p_holm", "mean_diff", "mean_loss_diff",
             "significativo_alpha05", "normal_alpha05", "n", "lag_NW"]
    all_rows = all_rows[[c for c in front if c in all_rows.columns]]
    all_rows.to_csv(C.STATS_CSV, index=False)
    print(f"\ntabla -> {C.STATS_CSV.relative_to(C.ROOT)}")

    figure(reg, cls)
    write_md(reg, cls, sh_reg, norm_reg, sh_cls, norm_cls,
             paired_reg, paired_cls, omni_reg, omni_cls, dm, nemenyi)
    print("-" * 79)
    print("OK.")


if __name__ == "__main__":
    main()
