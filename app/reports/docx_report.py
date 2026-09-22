"""Informe tecnico extenso (.docx): metodologia completa + resultados del escenario.

Combina valores en vivo de la base de datos con las tablas y resumenes generados
por los modulos de ml/ (EDA, entrenamiento, CV, tuning, pruebas estadisticas).
Cada figura y tabla va acompanada de un parrafo de interpretacion.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timezone

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from sqlalchemy.orm import Session

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


def _interp(doc: Document, text: str):
    p = doc.add_paragraph()
    r = p.add_run("Interpretación. ")
    r.bold = True
    r.font.color.rgb = _ACCENT
    p.add_run(text)
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.space_after = Pt(10)


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
    title = doc.add_heading("Gemelo Digital de Turberas — Informe técnico", level=0)
    title.runs[0].font.color.rgb = _INK
    doc.add_paragraph(
        f"Sitio: {site.name}"
        f"{f' ({site.fluxnet_id})' if site.fluxnet_id else ''} · "
        f"turbera {C.peatland_label(site.peatland_type)}"
    )
    doc.add_paragraph(f"Generado: {now:%Y-%m-%d %H:%M} UTC")
    rows = C.model_rows(db)
    winner = next((r for r in rows if r.is_active), None)
    hs = C.holdout_series(db, site, date(2018, 1, 1), date(2018, 12, 31))
    if winner:
        doc.add_paragraph(f"Modelo desplegado: {winner.name} · v{winner.version}")
    doc.add_paragraph(
        "Objetivo: predecir los flujos diarios de CO₂ (NEE) y CH₄ (FCH4) y la clase de "
        "balance de carbono (sumidero/fuente) de una turbera restaurada, y simular "
        "escenarios de nivel freático."
    )
    doc.add_page_break()

    # ============================================================ 1. Datos y ETL
    _h(doc, "1. Datos y ETL", 1)
    meta = None
    meta_path = C.PROCESSED / "DE-Zrk_etl_metadata.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc.add_paragraph(
        "Fuente: producto FLUXNET-CH4 del sitio DE-Zrk (Zarnekow, Alemania), un fen "
        "minerotrófico rehumedecido hidrológicamente en 2004–2005. Resolución diaria, "
        "2013–2018. El centinela de dato faltante (−9999) se convierte a NaN. Los objetivos "
        "de regresión son NEE_F_ANNOPTLM y FCH4_F_ANNOPTLM, ya rellenados al 100 % por el "
        "equipo del sitio con una red neuronal (ANNOPTLM); no se implementa gap-filling propio."
    )
    doc.add_paragraph(
        "El periodo de análisis se restringe a 2016–2018 porque el nivel freático (WTD) está "
        "100 % vacío en 2013–2015 —también su versión rellenada— y WTD es la variable de "
        "entrada obligatoria del simulador de escenarios. La clase de clasificación "
        "«clase_balance_carbono» se deriva del signo del NEE: sumidero si NEE < 0, fuente si "
        "NEE ≥ 0."
    )
    if meta:
        rc = meta.get("row_counts", {})
        _table(
            doc, ["Etapa", "Filas"],
            [
                ["CSV crudo (2013–2018)", rc.get("raw", "—")],
                ["Observaciones 2016–2018", rc.get("observations_2016_2018", "—")],
                ["Matriz ML (tras features y ventanas)", rc.get("ml_dataset", "—")],
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
    )

    # ============================================================ 2. EDA
    _h(doc, "2. Análisis exploratorio (EDA)", 1)
    desc = C.read_csv_rows(C.EDA_TBL / "01_descriptivos.csv")
    if desc:
        _table(
            doc,
            ["Variable", "media", "mediana", "desv. est.", "asimetría", "curtosis exc."],
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
    )
    norm = C.read_csv_rows(C.EDA_TBL / "06_normalidad.csv")
    if norm:
        _table(
            doc,
            ["Serie", "Shapiro W", "p-valor", "¿normal? (α=0.05)"],
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
        )
    _add_fig_path(doc, C.fig(C.EDA_FIG / "03_correlacion_pearson_heatmap.png"), 5.8)
    _interp(
        doc,
        "La radiación (SW_IN_F, NETRAD_F), la temperatura del aire y el VPD forman un bloque "
        "muy intercorrelacionado; TS_1…TS_5 son casi idénticas entre sí (justifica "
        "TS_promedio). El FCH4 correlaciona fuerte con la temperatura del suelo y con TA_F; "
        "el NEE, sobre todo con la radiación, con signo negativo (más luz → más fotosíntesis "
        "→ NEE más negativo).",
    )

    # ============================================================ 3. Entrenamiento
    _h(doc, "3. Entrenamiento y comparación de modelos", 1)
    doc.add_paragraph(
        "Se entrenan cinco modelos: Random Forest, Gradient Boosting (XGBoost) y SVR/SVM "
        "(clásicos), y CNN-LSTM y Stacking Ensemble (híbridos). Evaluación con holdout "
        "temporal: entrenamiento 2016–2017 (702 días), test 2018 (365 días), sin barajar. "
        "Para cada modelo se registra el tiempo de entrenamiento, y en regresión RMSE, MAE, "
        "R², NSE y KGE (para CO₂ y CH₄), y en clasificación accuracy, precision, recall, F1 "
        "(macro y por clase), matriz de confusión y AUC."
    )
    _table(
        doc,
        ["Modelo", "Familia", "RMSE", "MAE", "R²", "NSE", "KGE", "Acc.", "F1", "AUC", "t (s)"],
        [
            [f"{r.name}{' ★' if r.is_active else ''}",
             "híbrido" if r.algorithm_type == "HYBRID" else "clásico",
             C.fnum(r.rmse, 2), C.fnum(r.mae, 2), C.fnum(r.r2, 3), C.fnum(r.nse, 3),
             C.fnum(r.kge, 3), C.fnum(r.accuracy, 3), C.fnum(r.f1, 3), C.fnum(r.auc, 3),
             C.fnum(r.train_time_s, 1)]
            for r in rows
        ],
    )
    _note(doc, "★ modelo desplegado. RMSE/MAE/R²/NSE/KGE son la media de los objetivos CO₂ y CH₄; el KGE de NEE no está definido (media ≈ 0).")
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
    )

    # ============================================================ 4. Validación cruzada
    _h(doc, "4. Validación cruzada temporal", 1)
    doc.add_paragraph(
        f"Validación cruzada con TimeSeriesSplit de scikit-learn (nº de folds configurable; "
        f"valor usado en este informe: {folds}). Nunca KFold aleatorio: barajar el orden de "
        f"una serie temporal es fuga de datos. Cada fold amplía la ventana de entrenamiento y "
        f"desplaza la de test hacia adelante. Es complementaria al holdout único de la "
        f"sección 3 (registrado como fold = −1)."
    )
    cv = C.cv_summary_rows(db)
    if cv:
        _table(
            doc,
            ["Modelo", "folds", "RMSE (media±sd)", "R² (media±sd)", "F1 (media±sd)", "AUC (media±sd)"],
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
    )

    # ============================================================ 5. Tuning
    _h(doc, "5. Ajuste de hiperparámetros", 1)
    doc.add_paragraph(
        "Se ajustan Random Forest y Stacking Ensemble (los más estables en la CV) y "
        "opcionalmente XGBoost, con GridSearchCV / RandomizedSearchCV y TimeSeriesSplit(5) "
        "como validación interna. El ajuste se hace solo sobre el tramo de entrenamiento; el "
        "holdout 2018 queda intacto. Random Forest: n_estimators ∈ {200, 400, 800}, "
        "max_depth ∈ {None, 12, 24}, min_samples_leaf ∈ {1, 2, 4}. Meta-modelo del Stacking: "
        "LinearRegression vs Ridge(α ∈ {1, 10}) en regresión y LogisticRegression con C ∈ "
        "{0.1, 1, 10} en clasificación, más los hiperparámetros de los tres modelos base."
    )
    tuning_path = C.TRAIN_DIR / "modelo_final_metadata.json"
    if tuning_path.exists():
        import json

        tm = json.loads(tuning_path.read_text(encoding="utf-8"))
        bp = tm.get("best_params", {})
        if bp:
            _table(
                doc,
                ["Componente", "Mejores hiperparámetros"],
                [
                    ["Regresión", str(bp.get("regresion", "—"))],
                    ["Clasificación", str(bp.get("clasificacion", "—"))],
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
        )

    # ============================================================ 6. Pruebas estadísticas
    _h(doc, "6. Pruebas estadísticas de comparación de modelos", 1)
    doc.add_paragraph(
        "Sobre los cinco modelos base (sin tuning) y sus métricas por fold de la validación "
        "cruzada. Batería: Shapiro-Wilk (normalidad de las distribuciones de error entre "
        "folds), comparación pareada del mejor modelo (Stacking Ensemble) frente a cada uno "
        "de los demás (t pareada si normal, Wilcoxon signed-rank si no) con corrección de "
        "Holm, ómnibus ANOVA / Kruskal-Wallis más Friedman (medidas repetidas), post-hoc de "
        "Nemenyi tras Friedman, y prueba de Diebold-Mariano entre el mejor y el segundo "
        "mejor (Random Forest)."
    )
    st = C.read_csv_rows(C.TRAIN_DIR / "statistical_tests_resultados.csv")
    if st:
        keep = [
            r for r in st
            if r.get("seccion", "").startswith(("2-pareada-RMSE", "3-omnibus-RMSE", "3b-nemenyi", "4-diebold"))
        ]
        _table(
            doc,
            ["Sección", "Detalle", "Prueba", "p-valor", "¿signif. α=0.05?"],
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
    )

    # ============================================================ 7. Escenario simulado
    _h(doc, "7. Resultados del escenario simulado", 1)
    try:
        sim = C.scenario_prediction(db, scenario_id=scenario_id, wtd_cm=sim_wtd_cm, on_date=sim_date)
        _table(
            doc,
            ["Parámetro", "Valor"],
            [
                ["Nivel freático objetivo", f"{sim.target_water_table_depth_cm:.1f} cm"],
                ["Fecha", sim.on_date.isoformat()],
                ["Escenario", str(sim.scenario_id) if sim.scenario_id else "(directo)"],
                ["CO₂ predicho (NEE)", f"{sim.co2_predicted:+.3f} gC m⁻² d⁻¹"],
                ["CH₄ predicho (FCH4)", f"{sim.ch4_predicted:.2f} nmol m⁻² s⁻¹"],
                ["Clase", sim.predicted_class],
                ["P(sumidero)", f"{sim.proba_sumidero:.3f}"],
                ["Modelo", f"{sim.model_name} · {sim.model_version} ({sim.model_format})"],
            ],
        )
        doc.add_paragraph("Supuestos de la simulación:")
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
        )
    except Exception as exc:  # pragma: no cover
        doc.add_paragraph(f"No se pudo ejecutar el escenario: {exc}")

    # ============================================================ 7b. Matriz de confusión + ROC del ganador
    if winner and hs.available and hs.y_true:
        _h(doc, "7 bis. Matriz de confusión y ROC del modelo desplegado", 1)
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
        )

    # ============================================================ 8. Conclusiones
    _h(doc, "8. Conclusiones y limitaciones", 1)
    if winner:
        doc.add_paragraph(
            f"El modelo desplegado, {winner.name} (v{winner.version}), predice bien el flujo "
            f"de metano (R² ≈ {C.fnum(winner.r2, 2)} de media, AUC de clasificación "
            f"{C.fnum(winner.auc, 3)}) y de forma limitada el NEE diario. Es adecuado para "
            f"clasificar días sumidero/fuente y para comparar escenarios de nivel freático en "
            f"términos relativos."
        )
    for lim in [
        "El dataset no contiene índices de vegetación (NDVI, LAI); esto acota estructuralmente "
        "el R² del NEE. Trabajo futuro: incorporar productos de satélite o particionar GPP y "
        "RECO como objetivos intermedios.",
        "El sitio no midió humedad del suelo; la hidrología se representa solo con WTD y "
        "precipitación.",
        "El simulador asume nivel freático constante y meteo climatológica; no modela "
        "transitorios ni retroalimentaciones ecosistémicas.",
        "Solo DE-Zrk tiene un dataset procesado; para otros sitios el informe se limita a la "
        "metodología y la comparación de modelos.",
    ]:
        doc.add_paragraph(lim, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
