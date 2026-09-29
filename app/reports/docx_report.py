"""Informe tecnico extenso (.docx): metodologia completa + resultados del escenario.

Combina valores en vivo de la base de datos con las tablas y resumenes generados
por los modulos de ml/ (EDA, entrenamiento, CV, tuning, pruebas estadisticas).
Cada figura y tabla va acompanada de un parrafo de interpretacion. El idioma sale
de `app.core.i18n` (`force(language)` lo fija quien llama).
"""
from __future__ import annotations

import io
from datetime import date, datetime, timezone

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from sqlalchemy.orm import Session

from app.core.i18n import tr
from app.reports import common as C
from app.reports import figures as F

_INK = RGBColor(0x18, 0x18, 0x1B)
_ACCENT = RGBColor(0x04, 0x78, 0x57)


# --------------------------------------------------------------------- helpers
def _doc() -> Document:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)
    return doc


def _h(doc: Document, text: str, level: int = 1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = _INK
    return p


def _bloque_anotado(doc: Document, rotulo: str, text: str, espacio: int):
    p = doc.add_paragraph()
    r = p.add_run(f"{rotulo}. ")
    r.bold = True
    r.font.color.rgb = _ACCENT
    p.add_run(text)
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.space_after = Pt(espacio)


def _interp(doc: Document, es: str, en: str):
    """Que SIGNIFICA el dato. Va primero: es lo que sostiene el argumento."""
    _bloque_anotado(doc, tr("Interpretación", "Interpretation"), tr(es, en), espacio=2)


def _expl(doc: Document, es: str, en: str):
    """POR QUE se puede afirmar lo anterior: ejes, unidades, datos y calculo.

    Va debajo de la interpretacion, como respaldo auditable. El mismo orden que
    usa la aplicacion (`ui/interpret.py`), para que informe e interfaz se lean
    igual.
    """
    _bloque_anotado(doc, tr("Explicabilidad", "Explainability"), tr(es, en), espacio=10)


def _note(doc: Document, text: str):
    p = doc.add_paragraph(text)
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(9)


def _table(doc: Document, headers: list[str], rows: list[list[str]]):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, htext in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = str(htext)
        cell.paragraphs[0].runs[0].bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, val in enumerate(r):
            cells[i].text = str(val)
            for run in cells[i].paragraphs[0].runs:
                run.font.size = Pt(9)
    doc.add_paragraph()
    return t


def _add_fig(doc: Document, png: bytes, width_in: float = 6.2):
    from docx.shared import Inches

    doc.add_picture(io.BytesIO(png), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_fig_path(doc: Document, path, width_in: float = 6.2):
    from docx.shared import Inches

    if path and path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        return True
    return False


# --------------------------------------------------------------------- build
def build(
    db: Session,
    *,
    site_id: int,
    scenario_id: int | None = None,
    folds: int = 5,
    sim_wtd_cm: float | None = None,
    sim_date: date | None = None,
) -> bytes:
    site = C.get_site(db, site_id)
    sim_date = sim_date or date(2019, 7, 15)
    now = datetime.now(timezone.utc)
    doc = _doc()

    # ---------------------------------------------------------------- portada
    title = doc.add_heading(
        tr("Gemelo Digital de Turberas — Informe técnico",
           "Peatland Digital Twin — Technical report"), level=0)
    title.runs[0].font.color.rgb = _INK
    doc.add_paragraph(
        f"{tr('Sitio', 'Site')}: {site.name}"
        f"{f' ({site.fluxnet_id})' if site.fluxnet_id else ''} · "
        f"{tr('turbera', 'peatland')} {C.peatland_label(site.peatland_type)}"
    )
    doc.add_paragraph(f"{tr('Generado', 'Generated')}: {now:%Y-%m-%d %H:%M} UTC")
    rows = C.model_rows(db)
    winner = next((r for r in rows if r.is_active), None)
    hs = C.holdout_series(db, site, date(2018, 1, 1), date(2018, 12, 31))
    if winner:
        doc.add_paragraph(f"{tr('Modelo desplegado', 'Deployed model')}: {winner.name} · v{winner.version}")
    doc.add_paragraph(tr(
        "Objetivo: predecir los flujos diarios de CO₂ (NEE) y CH₄ (FCH4) y la clase de "
        "balance de carbono (sumidero/fuente) de una turbera restaurada, y simular "
        "escenarios de nivel freático.",
        "Goal: predict the daily CO₂ (NEE) and CH₄ (FCH4) fluxes and the carbon balance "
        "class (sink/source) of a restored peatland, and simulate water-table scenarios.",
    ))
    doc.add_page_break()

    # ============================================================ 1. Datos y ETL
    _h(doc, tr("1. Datos y ETL", "1. Data and ETL"), 1)
    meta = None
    meta_path = C.PROCESSED / "DE-Zrk_etl_metadata.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc.add_paragraph(tr(
        "Fuente: producto FLUXNET-CH4 del sitio DE-Zrk (Zarnekow, Alemania), un fen "
        "minerotrófico rehumedecido hidrológicamente en 2004–2005. Resolución diaria, "
        "2013–2018. El centinela de dato faltante (−9999) se convierte a NaN. Los objetivos "
        "de regresión son NEE_F_ANNOPTLM y FCH4_F_ANNOPTLM, ya rellenados al 100 % por el "
        "equipo del sitio con una red neuronal (ANNOPTLM); no se implementa gap-filling propio.",
        "Source: FLUXNET-CH4 product of the DE-Zrk site (Zarnekow, Germany), a minerotrophic "
        "fen hydrologically rewetted in 2004–2005. Daily resolution, 2013–2018. The "
        "missing-data sentinel (−9999) is converted to NaN. The regression targets are "
        "NEE_F_ANNOPTLM and FCH4_F_ANNOPTLM, already 100 % gap-filled by the site team with "
        "a neural network (ANNOPTLM); no gap-filling of our own is implemented.",
    ))
    doc.add_paragraph(tr(
        "El periodo de análisis se restringe a 2016–2018 porque el nivel freático (WTD) está "
        "100 % vacío en 2013–2015 —también su versión rellenada— y WTD es la variable de "
        "entrada obligatoria del simulador de escenarios. La clase de clasificación "
        "«clase_balance_carbono» se deriva del signo del NEE: sumidero si NEE < 0, fuente si "
        "NEE ≥ 0.",
        "The analysis period is restricted to 2016–2018 because the water table (WTD) is "
        "100 % empty in 2013–2015 —including its gap-filled version— and WTD is the "
        "mandatory input variable of the scenario simulator. The classification target "
        "«clase_balance_carbono» is derived from the sign of NEE: sink if NEE < 0, source "
        "if NEE ≥ 0.",
    ))
    if meta:
        rc = meta.get("row_counts", {})
        _table(
            doc, [tr("Etapa", "Stage"), tr("Filas", "Rows")],
            [
                [tr("CSV crudo (2013–2018)", "Raw CSV (2013–2018)"), rc.get("raw", "—")],
                [tr("Observaciones 2016–2018", "Observations 2016–2018"),
                 rc.get("observations_2016_2018", "—")],
                [tr("Matriz ML (tras features y ventanas)", "ML matrix (after features and windows)"),
                 rc.get("ml_dataset", "—")],
            ],
        )
    _interp(
        doc,
        "El filtrado temporal descarta la mitad del histórico pero es la única opción "
        "defendible: conservar 2013–2015 obligaría a imputar el 100 % del driver de interés. "
        "Las 32 features finales combinan meteorología gap-filled, WTD con sus retardos y "
        "medias móviles, la media del perfil de temperatura de suelo (TS_promedio, que "
        "sustituye a TS_1…TS_5 por su colinealidad r > 0.95) y variables estacionales. "
        "Limitación registrada: el sitio no midió humedad del suelo; no hay ninguna variable "
        "de contenido de agua del suelo en el dataset.",
        "The temporal filter discards half of the record but is the only defensible option: "
        "keeping 2013–2015 would require imputing 100 % of the driver of interest. The 32 "
        "final features combine gap-filled meteorology, WTD with its lags and moving "
        "averages, the mean of the soil temperature profile (TS_promedio, which replaces "
        "TS_1…TS_5 because of their collinearity r > 0.95) and seasonal variables. "
        "Recorded limitation: the site did not measure soil moisture; there is no soil "
        "water content variable in the dataset.",
    )
    _expl(
        doc,
        "La tabla cuenta filas en tres momentos del pipeline: el CSV crudo de "
        "FLUXNET-CH4 tal como se descarga, las observaciones que sobreviven al "
        "recorte temporal 2016-2018, y las de la matriz final de aprendizaje, ya "
        "con las ventanas y los retardos aplicados (que consumen los primeros días "
        "de la serie).",
        "The table counts rows at three points of the pipeline: the raw FLUXNET-CH4 CSV "
        "as downloaded, the observations that survive the 2016-2018 temporal cut, and "
        "those of the final learning matrix, with windows and lags already applied "
        "(which consume the first days of the series).",
    )

    # ============================================================ 2. EDA
    _h(doc, tr("2. Análisis exploratorio (EDA)", "2. Exploratory data analysis (EDA)"), 1)
    desc = C.read_csv_rows(C.EDA_TBL / "01_descriptivos.csv")
    if desc:
        _table(
            doc,
            [tr("Variable", "Variable"), tr("media", "mean"), tr("mediana", "median"),
             tr("desv. est.", "std. dev."), tr("asimetría", "skewness"),
             tr("curtosis exc.", "excess kurt.")],
            [
                [d.get("variable", ""), C.fnum(d.get("media"), 2), C.fnum(d.get("mediana"), 2),
                 C.fnum(d.get("desv_std"), 2), C.fnum(d.get("asimetria"), 2),
                 C.fnum(d.get("curtosis_exceso"), 2)]
                for d in desc
            ],
        )
    _add_fig_path(doc, C.fig(C.EDA_FIG / "01_histogramas_kde_variables_clave.png"), 6.4)
    _interp(
        doc,
        "El NEE tiene media ≈ 0 con desviación ≈ 1.1: el sitio es fuente de CO₂ la mayoría de "
        "los días con episodios intensos de captura en verano que tiran de la media hacia "
        "cero. El FCH4 es estrictamente positivo y muy asimétrico a la derecha (pulsos "
        "estivales) → conviene transformarlo antes de modelos que asuman normalidad. El WTD "
        "es predominantemente positivo (lámina de agua sobre la superficie): confirma que "
        "Zarnekow opera como fen inundado. El perfil de temperatura del suelo amortigua su "
        "amplitud con la profundidad, de ahí que baste TS_promedio.",
        "NEE has a mean ≈ 0 with a deviation ≈ 1.1: the site is a CO₂ source most days, with "
        "intense summer uptake episodes pulling the mean towards zero. FCH4 is strictly "
        "positive and strongly right-skewed (summer pulses) → it should be transformed "
        "before models that assume normality. WTD is predominantly positive (water layer "
        "above the surface): it confirms that Zarnekow behaves as a flooded fen. The soil "
        "temperature profile damps its amplitude with depth, hence TS_promedio suffices.",
    )
    _expl(
        doc,
        "Los estadísticos se calculan sobre los 1096 días de la serie diaria. Media "
        "y mediana van en la unidad de cada variable: NEE en gC m⁻² d⁻¹, FCH4 en "
        "nmol m⁻² s⁻¹, temperaturas en °C, WTD en metros sobre la superficie. La "
        "asimetría y la curtosis de exceso valen 0 en una distribución normal, así "
        "que cuánto más se alejan de 0, más se aparta la variable de ella. La "
        "figura muestra el histograma de cada variable con su estimación de "
        "densidad superpuesta.",
        "The statistics are computed over the 1096 days of the daily series. Mean and "
        "median are in each variable's unit: NEE in gC m⁻² d⁻¹, FCH4 in nmol m⁻² s⁻¹, "
        "temperatures in °C, WTD in meters above the surface. Skewness and excess "
        "kurtosis are 0 for a normal distribution, so the further they are from 0, the "
        "more the variable departs from it. The figure shows each variable's histogram "
        "with its density estimate overlaid.",
    )
    norm = C.read_csv_rows(C.EDA_TBL / "06_normalidad.csv")
    if norm:
        _table(
            doc,
            [tr("Serie", "Series"), "Shapiro W", tr("p-valor", "p-value"),
             tr("¿normal? (α=0.05)", "normal? (α=0.05)")],
            [
                [n.get("serie", ""), C.fnum(n.get("shapiro_W"), 4), C.fnum(n.get("shapiro_p"), 4),
                 n.get("normal_shapiro_5pct", "")]
                for n in norm
            ],
        )
        _interp(
            doc,
            "Ninguna variable clave ni los residuos de la descomposición estacional superan "
            "Shapiro-Wilk: se rechaza la normalidad en todos los casos (p < 0.05). Esto "
            "condiciona el paso de pruebas estadísticas: se usan contrastes no paramétricos "
            "(Wilcoxon, Kruskal-Wallis, Friedman) o transformaciones (log1p en FCH4).",
            "No key variable nor the residuals of the seasonal decomposition pass "
            "Shapiro-Wilk: normality is rejected in all cases (p < 0.05). This conditions "
            "the statistical-testing step: non-parametric tests (Wilcoxon, Kruskal-Wallis, "
            "Friedman) or transformations (log1p on FCH4) are used.",
        )
        _expl(
            doc,
            "Shapiro-Wilk contrasta la hipótesis nula de que los datos proceden de una "
            "distribución normal: el estadístico W se acerca a 1 cuando lo son, y un "
            "p-valor por debajo de 0.05 RECHAZA esa hipótesis. Se aplica a las "
            "variables clave y también a los residuos de la descomposición estacional, "
            "para separar la no normalidad propia de la variable de la que induce el "
            "ciclo anual.",
            "Shapiro-Wilk tests the null hypothesis that the data come from a normal "
            "distribution: the W statistic approaches 1 when they do, and a p-value below "
            "0.05 REJECTS that hypothesis. It is applied to the key variables and also to "
            "the residuals of the seasonal decomposition, to separate the variable's own "
            "non-normality from that induced by the annual cycle.",
        )
    _add_fig_path(doc, C.fig(C.EDA_FIG / "03_correlacion_pearson_heatmap.png"), 5.8)
    _interp(
        doc,
        "La radiación (SW_IN_F, NETRAD_F), la temperatura del aire y el VPD forman un bloque "
        "muy intercorrelacionado; TS_1…TS_5 son casi idénticas entre sí (justifica "
        "TS_promedio). El FCH4 correlaciona fuerte con la temperatura del suelo y con TA_F; "
        "el NEE, sobre todo con la radiación, con signo negativo (más luz → más fotosíntesis "
        "→ NEE más negativo).",
        "Radiation (SW_IN_F, NETRAD_F), air temperature and VPD form a highly "
        "intercorrelated block; TS_1…TS_5 are almost identical to each other (which "
        "justifies TS_promedio). FCH4 correlates strongly with soil temperature and TA_F; "
        "NEE mainly with radiation, with a negative sign (more light → more photosynthesis "
        "→ more negative NEE).",
    )
    _expl(
        doc,
        "Cada celda del mapa es el coeficiente de correlación de Pearson del par "
        "fila-columna, entre −1 (relación inversa perfecta) y +1 (directa "
        "perfecta), con el 0 en el centro de la escala de color. Pearson mide "
        "relación LINEAL; la diagonal vale 1 por definición y la matriz es "
        "simétrica.",
        "Each cell of the map is the Pearson correlation coefficient of the row-column "
        "pair, between −1 (perfect inverse relation) and +1 (perfect direct), with 0 at "
        "the center of the color scale. Pearson measures LINEAR relation; the diagonal is "
        "1 by definition and the matrix is symmetric.",
    )

    # ============================================================ 3. Entrenamiento
    _h(doc, tr("3. Entrenamiento y comparación de modelos", "3. Training and model comparison"), 1)
    doc.add_paragraph(tr(
        "Se entrenan cinco modelos: Random Forest, Gradient Boosting (XGBoost) y SVR/SVM "
        "(clásicos), y CNN-LSTM y Stacking Ensemble (híbridos). Evaluación con holdout "
        "temporal: entrenamiento 2016–2017 (702 días), test 2018 (365 días), sin barajar. "
        "Para cada modelo se registra el tiempo de entrenamiento, y en regresión RMSE, MAE, "
        "R², NSE y KGE (para CO₂ y CH₄), y en clasificación accuracy, precision, recall, F1 "
        "(macro y por clase), matriz de confusión y AUC.",
        "Five models are trained: Random Forest, Gradient Boosting (XGBoost) and SVR/SVM "
        "(classical), and CNN-LSTM and Stacking Ensemble (hybrid). Evaluation with a "
        "temporal holdout: training 2016–2017 (702 days), test 2018 (365 days), without "
        "shuffling. For each model the training time is recorded, and in regression RMSE, "
        "MAE, R², NSE and KGE (for CO₂ and CH₄), and in classification accuracy, precision, "
        "recall, F1 (macro and per class), confusion matrix and AUC.",
    ))
    _table(
        doc,
        [tr("Modelo", "Model"), tr("Familia", "Family"), "RMSE", "MAE", "R²", "NSE", "KGE",
         "Acc.", "F1", "AUC", "t (s)"],
        [
            [f"{r.name}{' ★' if r.is_active else ''}",
             tr("híbrido", "hybrid") if r.algorithm_type == "HYBRID" else tr("clásico", "classical"),
             C.fnum(r.rmse, 2), C.fnum(r.mae, 2), C.fnum(r.r2, 3), C.fnum(r.nse, 3),
             C.fnum(r.kge, 3), C.fnum(r.accuracy, 3), C.fnum(r.f1, 3), C.fnum(r.auc, 3),
             C.fnum(r.train_time_s, 1)]
            for r in rows
        ],
    )
    _note(doc, tr(
        "★ modelo desplegado. RMSE/MAE/R²/NSE/KGE son la media de los objetivos CO₂ y CH₄; el KGE de NEE no está definido (media ≈ 0).",
        "★ deployed model. RMSE/MAE/R²/NSE/KGE are the mean of the CO₂ and CH₄ targets; the NEE KGE is not defined (mean ≈ 0).",
    ))
    _add_fig_path(doc, C.fig(C.TRAIN_FIG / "heatmap_comparativo_metricas.png"), 6.4)
    _interp(
        doc,
        "El criterio de selección es combined = 0.5·R²norm + 0.5·(0.5·F1_macro + 0.5·AUC), "
        "50 % regresión / 50 % clasificación; en clasificación se usan F1_macro y AUC (no "
        "accuracy) porque con 32 % de sumidero un clasificador «siempre fuente» ya obtiene "
        "≈ 0.68 de accuracy sin aprender nada. El R² del NEE es bajo en los cinco modelos "
        "(≈ 0.15–0.24): NEE = GPP − RECO es la pequeña diferencia entre dos flujos grandes y "
        "depende del estado de la vegetación, ausente en el dataset (sin NDVI/LAI); un R² "
        "≈ 0.2 para NEE diario es coherente con la literatura de flujos en turberas. El CH₄, "
        "controlado por temperatura de suelo y nivel freático (ambos medidos), se predice "
        "bien (R² ≈ 0.7–0.85).",
        "The selection criterion is combined = 0.5·R²norm + 0.5·(0.5·F1_macro + 0.5·AUC), "
        "50 % regression / 50 % classification; in classification F1_macro and AUC are used "
        "(not accuracy) because with 32 % sink an «always source» classifier already gets "
        "≈ 0.68 accuracy without learning anything. The NEE R² is low in all five models "
        "(≈ 0.15–0.24): NEE = GPP − RECO is the small difference between two large fluxes "
        "and depends on the vegetation state, absent from the dataset (no NDVI/LAI); an R² "
        "≈ 0.2 for daily NEE is consistent with the peatland flux literature. CH₄, "
        "controlled by soil temperature and water table (both measured), is predicted well "
        "(R² ≈ 0.7–0.85).",
    )
    _expl(
        doc,
        "Todas las métricas proceden del mismo holdout temporal: se entrena con "
        "2016-2017 y se evalúa sobre 2018 completo, sin barajar. RMSE y MAE son "
        "errores en la unidad del objetivo, luego menor es mejor; R², NSE y KGE son "
        "adimensionales y valen 1 en la predicción perfecta, 0 cuando el modelo "
        "iguala a predecir la media y negativo si la empeora. Accuracy, F1 y AUC "
        "corresponden a la clasificación sumidero/fuente. En el mapa de calor cada "
        "columna se normaliza min-max por separado, de modo que el color indica el "
        "ORDEN dentro de esa métrica y no su magnitud absoluta.",
        "All metrics come from the same temporal holdout: trained on 2016-2017 and "
        "evaluated on the whole of 2018, without shuffling. RMSE and MAE are errors in the "
        "target's unit, so lower is better; R², NSE and KGE are dimensionless and equal 1 "
        "for the perfect prediction, 0 when the model matches predicting the mean and "
        "negative if it does worse. Accuracy, F1 and AUC correspond to the sink/source "
        "classification. In the heat map each column is min-max normalized separately, so "
        "the color shows the RANK within that metric and not its absolute magnitude.",
    )

    # ============================================================ 4. Validación cruzada
    _h(doc, tr("4. Validación cruzada temporal", "4. Temporal cross-validation"), 1)
    doc.add_paragraph(tr(
        f"Validación cruzada con TimeSeriesSplit de scikit-learn (nº de folds configurable; "
        f"valor usado en este informe: {folds}). Nunca KFold aleatorio: barajar el orden de "
        f"una serie temporal es fuga de datos. Cada fold amplía la ventana de entrenamiento y "
        f"desplaza la de test hacia adelante. Es complementaria al holdout único de la "
        f"sección 3 (registrado como fold = −1).",
        f"Cross-validation with scikit-learn's TimeSeriesSplit (configurable number of "
        f"folds; value used in this report: {folds}). Never random KFold: shuffling the "
        f"order of a time series is data leakage. Each fold widens the training window and "
        f"shifts the test window forward. It complements the single holdout of section 3 "
        f"(recorded as fold = −1).",
    ))
    cv = C.cv_summary_rows(db)
    if cv:
        ms = tr("media±sd", "mean±sd")
        _table(
            doc,
            [tr("Modelo", "Model"), "folds", f"RMSE ({ms})", f"R² ({ms})", f"F1 ({ms})", f"AUC ({ms})"],
            [
                [
                    r["modelo"], r["n_folds"],
                    f"{C.fnum(r['rmse_mean'], 2)} ± {C.fnum(r['rmse_std'], 2)}",
                    f"{C.fnum(r['r2_mean'], 3)} ± {C.fnum(r['r2_std'], 3)}",
                    f"{C.fnum(r['f1_mean'], 3)} ± {C.fnum(r['f1_std'], 3)}",
                    f"{C.fnum(r['auc_mean'], 3)} ± {C.fnum(r['auc_std'], 3)}",
                ]
                for r in cv
            ],
        )
    _add_fig_path(doc, C.fig(C.TRAIN_FIG / "cv_boxplots_estabilidad.png"), 6.4)
    _interp(
        doc,
        "La desviación estándar entre folds mide la estabilidad de cada modelo. Random Forest "
        "y Stacking Ensemble son los más consistentes (menor sd de R² en FCH4); SVR/SVM y "
        "CNN-LSTM, los que más varían. El primer fold entrena con muy pocos días (< medio "
        "ciclo estacional) y da R² negativos en casi todos los modelos: es un hallazgo "
        "esperado sobre suficiencia de datos, no un fallo. NSE ≡ R² por definición.",
        "The standard deviation across folds measures each model's stability. Random Forest "
        "and Stacking Ensemble are the most consistent (lowest R² sd on FCH4); SVR/SVM and "
        "CNN-LSTM vary the most. The first fold trains on very few days (< half a seasonal "
        "cycle) and yields negative R² in almost all models: an expected finding about data "
        "sufficiency, not a failure. NSE ≡ R² by definition.",
    )
    _expl(
        doc,
        "Cada fila resume los folds de `TimeSeriesSplit`: la media dice el "
        "rendimiento típico y la desviación típica, cuánto varía de un periodo a "
        "otro. Cada fold amplía la ventana de entrenamiento y desplaza la de test "
        "hacia adelante, de forma que el modelo nunca ve datos posteriores a los "
        "que predice. En la figura, cada caja abarca del primer al tercer cuartil "
        "de esos folds, con la mediana dentro y los bigotes en el mínimo y el "
        "máximo.",
        "Each row summarizes the `TimeSeriesSplit` folds: the mean gives the typical "
        "performance and the standard deviation how much it varies from one period to "
        "another. Each fold widens the training window and shifts the test window forward, "
        "so the model never sees data later than what it predicts. In the figure, each box "
        "spans the first to the third quartile of those folds, with the median inside and "
        "the whiskers at the minimum and maximum.",
    )

    # ============================================================ 5. Tuning
    _h(doc, tr("5. Ajuste de hiperparámetros", "5. Hyperparameter tuning"), 1)
    doc.add_paragraph(tr(
        "Se ajustan Random Forest y Stacking Ensemble (los más estables en la CV) y "
        "opcionalmente XGBoost, con GridSearchCV / RandomizedSearchCV y TimeSeriesSplit(5) "
        "como validación interna. El ajuste se hace solo sobre el tramo de entrenamiento; el "
        "holdout 2018 queda intacto. Random Forest: n_estimators ∈ {200, 400, 800}, "
        "max_depth ∈ {None, 12, 24}, min_samples_leaf ∈ {1, 2, 4}. Meta-modelo del Stacking: "
        "LinearRegression vs Ridge(α ∈ {1, 10}) en regresión y LogisticRegression con C ∈ "
        "{0.1, 1, 10} en clasificación, más los hiperparámetros de los tres modelos base.",
        "Random Forest and Stacking Ensemble (the most stable in CV) and optionally XGBoost "
        "are tuned, with GridSearchCV / RandomizedSearchCV and TimeSeriesSplit(5) as "
        "internal validation. Tuning is done only on the training segment; the 2018 holdout "
        "remains untouched. Random Forest: n_estimators ∈ {200, 400, 800}, max_depth ∈ "
        "{None, 12, 24}, min_samples_leaf ∈ {1, 2, 4}. Stacking meta-model: "
        "LinearRegression vs Ridge(α ∈ {1, 10}) in regression and LogisticRegression with "
        "C ∈ {0.1, 1, 10} in classification, plus the hyperparameters of the three base "
        "models.",
    ))
    tuning_path = C.TRAIN_DIR / "modelo_final_metadata.json"
    if tuning_path.exists():
        import json

        tm = json.loads(tuning_path.read_text(encoding="utf-8"))
        bp = tm.get("best_params", {})
        if bp:
            _table(
                doc,
                [tr("Componente", "Component"), tr("Mejores hiperparámetros", "Best hyperparameters")],
                [
                    [tr("Regresión", "Regression"), str(bp.get("regresion", "—"))],
                    [tr("Clasificación", "Classification"), str(bp.get("clasificacion", "—"))],
                ],
            )
        _interp(
            doc,
            f"El modelo tuneado ({tm.get('winner', 'Stacking Ensemble (tuned)')}) mejora el "
            f"criterio combinado respecto a su versión base (≈ 0.907 → "
            f"{C.fnum(tm.get('combined_score'), 3)}). La mejora clave es el meta-modelo "
            f"Ridge(α = 10) en la regresión del stacking: la regularización L2 sobre las "
            f"predicciones base —ruidosas para el NEE— sube el R² del NEE de ≈ 0.216 a "
            f"≈ 0.236. El F1 y el AUC quedan prácticamente iguales. Este modelo tuneado es el "
            f"que queda desplegado (is_active = true).",
            f"The tuned model ({tm.get('winner', 'Stacking Ensemble (tuned)')}) improves the "
            f"combined criterion over its base version (≈ 0.907 → "
            f"{C.fnum(tm.get('combined_score'), 3)}). The key improvement is the Ridge(α = 10) "
            f"meta-model in the stacking regression: L2 regularization over the base "
            f"predictions —noisy for NEE— raises the NEE R² from ≈ 0.216 to ≈ 0.236. F1 and "
            f"AUC remain practically the same. This tuned model is the one left deployed "
            f"(is_active = true).",
        )
        _expl(
            doc,
            "Los hiperparámetros se buscan con validación interna TimeSeriesSplit(5) "
            "aplicada SOLO al tramo de entrenamiento, de modo que el holdout de 2018 no "
            "interviene en la búsqueda y las métricas de la sección anterior siguen "
            "siendo válidas. La tabla recoge los valores ganadores, no el espacio "
            "explorado.",
            "Hyperparameters are searched with TimeSeriesSplit(5) internal validation "
            "applied ONLY to the training segment, so the 2018 holdout takes no part in the "
            "search and the metrics of the previous section remain valid. The table lists "
            "the winning values, not the explored space.",
        )

    # ============================================================ 6. Pruebas estadísticas
    _h(doc, tr("6. Pruebas estadísticas de comparación de modelos",
               "6. Statistical tests for model comparison"), 1)
    doc.add_paragraph(tr(
        "Sobre los cinco modelos base (sin tuning) y sus métricas por fold de la validación "
        "cruzada. Batería: Shapiro-Wilk (normalidad de las distribuciones de error entre "
        "folds), comparación pareada del mejor modelo (Stacking Ensemble) frente a cada uno "
        "de los demás (t pareada si normal, Wilcoxon signed-rank si no) con corrección de "
        "Holm, ómnibus ANOVA / Kruskal-Wallis más Friedman (medidas repetidas), post-hoc de "
        "Nemenyi tras Friedman, y prueba de Diebold-Mariano entre el mejor y el segundo "
        "mejor (Random Forest).",
        "On the five base models (without tuning) and their per-fold cross-validation "
        "metrics. Battery: Shapiro-Wilk (normality of the error distributions across "
        "folds), paired comparison of the best model (Stacking Ensemble) against each of "
        "the others (paired t if normal, Wilcoxon signed-rank if not) with Holm correction, "
        "omnibus ANOVA / Kruskal-Wallis plus Friedman (repeated measures), Nemenyi post-hoc "
        "after Friedman, and a Diebold-Mariano test between the best and the second best "
        "(Random Forest).",
    ))
    st = C.read_csv_rows(C.TRAIN_DIR / "statistical_tests_resultados.csv")
    if st:
        keep = [
            r for r in st
            if r.get("seccion", "").startswith(("2-pareada-RMSE", "3-omnibus-RMSE", "3b-nemenyi", "4-diebold"))
        ]
        _table(
            doc,
            [tr("Sección", "Section"), tr("Detalle", "Detail"), tr("Prueba", "Test"),
             tr("p-valor", "p-value"), tr("¿signif. α=0.05?", "signif. α=0.05?")],
            [
                [r.get("seccion", ""), r.get("detalle", ""), r.get("test", ""),
                 C.fnum(r.get("p_value"), 4), r.get("significativo_alpha05", "")]
                for r in keep
            ],
        )
    _add_fig_path(doc, C.fig(C.TRAIN_FIG / "statistical_tests_nemenyi_cd.png"), 5.6)
    _interp(
        doc,
        "Shapiro rechaza la normalidad → pruebas no paramétricas. Ninguna comparación pareada "
        "(Wilcoxon + Holm) alcanza p < 0.05: con 5 folds Wilcoxon no puede bajar de p ≈ "
        "0.0625, ni siquiera frente a SVR o CNN-LSTM, claramente peores. El ómnibus "
        "Kruskal-Wallis no detecta diferencias (p ≈ 0.26), pero Friedman —que respeta el "
        "bloqueo por fold— sí (p ≈ 0.0037). El post-hoc de Nemenyi aísla los pares "
        "responsables: CNN-LSTM es significativamente peor que Random Forest (p ≈ 0.041) y "
        "que XGBoost (p ≈ 0.023); entre Stacking, Random Forest y XGBoost no hay ninguna "
        "diferencia significativa en RMSE (p ≈ 1.0). Diebold-Mariano entre Stacking y Random "
        "Forest sobre el holdout 2018 tampoco encuentra diferencia significativa (p ≈ 0.75 "
        "para el NEE, ≈ 0.14 para el CH₄), aunque el signo favorece al Stacking. Conclusión: "
        "la elección del Stacking no se sostiene en un RMSE mejor —es estadísticamente "
        "indistinguible de RF/XGBoost— sino en su mejor F1_macro, AUC y estabilidad. Nota: "
        "estas pruebas se hicieron sobre los cinco modelos base; el modelo desplegado "
        "(Stacking tuned) es un refinamiento posterior del ganador ya seleccionado.",
        "Shapiro rejects normality → non-parametric tests. No paired comparison (Wilcoxon + "
        "Holm) reaches p < 0.05: with 5 folds Wilcoxon cannot go below p ≈ 0.0625, not even "
        "against SVR or CNN-LSTM, clearly worse. The Kruskal-Wallis omnibus detects no "
        "differences (p ≈ 0.26), but Friedman —which respects the blocking by fold— does "
        "(p ≈ 0.0037). The Nemenyi post-hoc isolates the responsible pairs: CNN-LSTM is "
        "significantly worse than Random Forest (p ≈ 0.041) and than XGBoost (p ≈ 0.023); "
        "among Stacking, Random Forest and XGBoost there is no significant RMSE difference "
        "(p ≈ 1.0). Diebold-Mariano between Stacking and Random Forest on the 2018 holdout "
        "also finds no significant difference (p ≈ 0.75 for NEE, ≈ 0.14 for CH₄), although "
        "the sign favors the Stacking. Conclusion: the choice of the Stacking does not rest "
        "on a better RMSE —it is statistically indistinguishable from RF/XGBoost— but on "
        "its better F1_macro, AUC and stability. Note: these tests were run on the five "
        "base models; the deployed model (tuned Stacking) is a later refinement of the "
        "already selected winner.",
    )
    _expl(
        doc,
        "Los contrastes operan sobre las métricas por fold de la validación "
        "cruzada, con los cinco modelos base y α = 0.05. Wilcoxon compara pares "
        "emparejando fold a fold; Friedman pregunta si algún modelo difiere del "
        "conjunto respetando ese emparejamiento; Nemenyi identifica después que "
        "pares concretos lo explican. En el diagrama, los modelos se sitúan por "
        "rango medio (1 = mejor) y una barra horizontal une a los que quedan a "
        "menos de la diferencia crítica CD, es decir, a los que NO se pueden "
        "distinguir.",
        "The tests operate on the per-fold cross-validation metrics, with the five base "
        "models and α = 0.05. Wilcoxon compares pairs, pairing fold by fold; Friedman asks "
        "whether any model differs from the set while respecting that pairing; Nemenyi then "
        "identifies which specific pairs explain it. In the diagram, models are placed by "
        "mean rank (1 = best) and a horizontal bar joins those that lie within the "
        "critical difference CD, that is, those that CANNOT be told apart.",
    )

    # ============================================================ 7. Escenario simulado
    _h(doc, tr("7. Resultados del escenario simulado", "7. Simulated scenario results"), 1)
    try:
        sim = C.scenario_prediction(db, scenario_id=scenario_id, wtd_cm=sim_wtd_cm, on_date=sim_date)
        clase = tr(
            {"sumidero": "sumidero", "fuente": "fuente"}.get(str(sim.predicted_class), str(sim.predicted_class)),
            {"sumidero": "sink", "fuente": "source"}.get(str(sim.predicted_class), str(sim.predicted_class)),
        )
        _table(
            doc,
            [tr("Parámetro", "Parameter"), tr("Valor", "Value")],
            [
                [tr("Nivel freático objetivo", "Target water table"),
                 f"{sim.target_water_table_depth_cm:.1f} cm"],
                [tr("Fecha", "Date"), sim.on_date.isoformat()],
                [tr("Escenario", "Scenario"),
                 str(sim.scenario_id) if sim.scenario_id else tr("(directo)", "(direct)")],
                [tr("CO₂ predicho (NEE)", "Predicted CO₂ (NEE)"),
                 f"{sim.co2_predicted:+.3f} gC m⁻² d⁻¹"],
                [tr("CH₄ predicho (FCH4)", "Predicted CH₄ (FCH4)"),
                 f"{sim.ch4_predicted:.2f} nmol m⁻² s⁻¹"],
                [tr("Clase", "Class"), clase],
                ["P(" + tr("sumidero", "sink") + ")", f"{sim.proba_sumidero:.3f}"],
                [tr("Modelo", "Model"), f"{sim.model_name} · {sim.model_version} ({sim.model_format})"],
            ],
        )
        doc.add_paragraph(tr("Supuestos de la simulación:", "Simulation assumptions:"))
        for a in sim.assumptions:
            doc.add_paragraph(a, style="List Bullet")
        _interp(
            doc,
            f"Con el nivel freático llevado a {sim.target_water_table_depth_cm:.0f} cm y el "
            f"resto de condiciones en su climatología del día del año, el modelo predice un "
            f"balance de tipo «{sim.predicted_class}» (NEE = "
            f"{sim.co2_predicted:+.2f} gC m⁻² d⁻¹) con una emisión de metano de "
            f"{sim.ch4_predicted:.0f} nmol m⁻² s⁻¹. La incertidumbre del NEE es alta (R² de "
            f"NEE en holdout ≈ 0.24), por lo que el valor puntual debe leerse como una "
            f"tendencia, no como una cifra exacta; la clase y el orden de magnitud del CH₄ "
            f"son más fiables.",
            f"With the water table set to {sim.target_water_table_depth_cm:.0f} cm and the "
            f"remaining conditions at their day-of-year climatology, the model predicts a "
            f"«{clase}» balance (NEE = {sim.co2_predicted:+.2f} gC m⁻² d⁻¹) with a methane "
            f"emission of {sim.ch4_predicted:.0f} nmol m⁻² s⁻¹. The NEE uncertainty is high "
            f"(holdout NEE R² ≈ 0.24), so the point value should be read as a trend, not as "
            f"an exact figure; the class and the order of magnitude of CH₄ are more reliable.",
        )
        _expl(
            doc,
            "El escenario fija el nivel freático objetivo y toma el resto de drivers "
            "—meteorología y temperatura de suelo— de la climatología del día del año "
            "correspondiente, promediada sobre 2016–2018. Por tanto es una simulación "
            "hipotética, no la reconstrucción de un día real, y mueve una sola variable "
            "dejando las demás congeladas.",
            "The scenario fixes the target water table and takes the remaining drivers "
            "—meteorology and soil temperature— from the climatology of the corresponding "
            "day of the year, averaged over 2016–2018. It is therefore a hypothetical "
            "simulation, not the reconstruction of a real day, and it moves a single "
            "variable while keeping the others frozen.",
        )
    except Exception as exc:  # pragma: no cover
        doc.add_paragraph(tr(f"No se pudo ejecutar el escenario: {exc}",
                             f"The scenario could not be run: {exc}"))

    # ============================================================ 7b. Matriz de confusión + ROC del ganador
    if winner and hs.available and hs.y_true:
        _h(doc, tr("7 bis. Matriz de confusión y ROC del modelo desplegado",
                   "7b. Confusion matrix and ROC of the deployed model"), 1)
        tn, fp, fn, tp = C.confusion_from_series(hs)
        _add_fig(doc, F.confusion_png(tn, fp, fn, tp, winner.name), 3.6)
        _add_fig(doc, F.roc_png(hs.y_true, hs.proba, winner.auc, winner.name), 4.0)
        total = tn + fp + fn + tp
        rec_sink = tp / (tp + fn) if (tp + fn) else 0
        _interp(
            doc,
            f"En el holdout 2018 ({total} días) el modelo acierta {tn + tp} "
            f"(accuracy = {C.fnum(winner.accuracy, 3)}) y detecta {tp} de {tp + fn} días de "
            f"sumidero (recall = {rec_sink:.2f}). El AUC = {C.fnum(winner.auc, 3)} confirma "
            f"una buena separación entre clases; el umbral 0.5 es el punto de operación por "
            f"defecto.",
            f"On the 2018 holdout ({total} days) the model gets {tn + tp} right "
            f"(accuracy = {C.fnum(winner.accuracy, 3)}) and detects {tp} of {tp + fn} sink "
            f"days (recall = {rec_sink:.2f}). The AUC = {C.fnum(winner.auc, 3)} confirms a "
            f"good separation between classes; the 0.5 threshold is the default operating "
            f"point.",
        )
        _expl(
            doc,
            "En la matriz, las filas son la clase real y las columnas la predicha, así "
            "que la diagonal recoge los aciertos; la clase positiva es «sumidero», "
            "definida como NEE < 0, y es la minoritaria. La curva ROC enfrenta la tasa "
            "de falsos positivos a la de verdaderos positivos recorriendo todos los "
            "umbrales de decisión posibles: la diagonal representa al clasificador "
            "aleatorio y el AUC es el área bajo la curva, de modo que mide la capacidad "
            "de ORDENAR los días por probabilidad de sumidero y no el acierto con un "
            "umbral concreto.",
            "In the matrix, rows are the true class and columns the predicted one, so the "
            "diagonal holds the hits; the positive class is «sink», defined as NEE < 0, and "
            "is the minority. The ROC curve plots the false positive rate against the true "
            "positive rate sweeping all possible decision thresholds: the diagonal "
            "represents the random classifier and the AUC is the area under the curve, so it "
            "measures the ability to RANK days by sink probability and not the accuracy at a "
            "specific threshold.",
        )

    # ============================================================ 8. Conclusiones
    _h(doc, tr("8. Conclusiones y limitaciones", "8. Conclusions and limitations"), 1)
    if winner:
        doc.add_paragraph(tr(
            f"El modelo desplegado, {winner.name} (v{winner.version}), predice bien el flujo "
            f"de metano (R² ≈ {C.fnum(winner.r2, 2)} de media, AUC de clasificación "
            f"{C.fnum(winner.auc, 3)}) y de forma limitada el NEE diario. Es adecuado para "
            f"clasificar días sumidero/fuente y para comparar escenarios de nivel freático en "
            f"términos relativos.",
            f"The deployed model, {winner.name} (v{winner.version}), predicts the methane "
            f"flux well (R² ≈ {C.fnum(winner.r2, 2)} on average, classification AUC "
            f"{C.fnum(winner.auc, 3)}) and the daily NEE only to a limited extent. It is "
            f"suitable for classifying sink/source days and for comparing water-table "
            f"scenarios in relative terms.",
        ))
    for lim in [
        tr("El dataset no contiene índices de vegetación (NDVI, LAI); esto acota estructuralmente "
           "el R² del NEE. Trabajo futuro: incorporar productos de satélite o particionar GPP y "
           "RECO como objetivos intermedios.",
           "The dataset contains no vegetation indices (NDVI, LAI); this structurally bounds "
           "the NEE R². Future work: incorporate satellite products or partition GPP and RECO "
           "as intermediate targets."),
        tr("El sitio no midió humedad del suelo; la hidrología se representa solo con WTD y "
           "precipitación.",
           "The site did not measure soil moisture; hydrology is represented only by WTD and "
           "precipitation."),
        tr("El simulador asume nivel freático constante y meteo climatológica; no modela "
           "transitorios ni retroalimentaciones ecosistémicas.",
           "The simulator assumes a constant water table and climatological weather; it does "
           "not model transients or ecosystem feedbacks."),
        tr("Solo DE-Zrk tiene un dataset procesado; para otros sitios el informe se limita a la "
           "metodología y la comparación de modelos.",
           "Only DE-Zrk has a processed dataset; for other sites the report is limited to the "
           "methodology and the model comparison."),
    ]:
        doc.add_paragraph(lim, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
