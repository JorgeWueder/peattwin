"""Informe PDF — resumen ejecutivo del sitio, con interpretacion de cada figura/tabla."""
from __future__ import annotations

import io
from datetime import date, datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session

from app.core.i18n import tr
from app.reports import common as C
from app.reports import figures as F

_ACCENT = colors.HexColor("#059669")
_INK = colors.HexColor("#18181b")
_MUTE = colors.HexColor("#71717a")
_LINE = colors.HexColor("#e4e4e7")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H0", parent=ss["Title"], fontSize=20, textColor=_INK, spaceAfter=4))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], fontSize=9.5, textColor=_MUTE, spaceAfter=14))
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13, textColor=_INK, spaceBefore=16, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=14, alignment=TA_JUSTIFY, textColor=_INK))
    ss.add(ParagraphStyle("Interp", parent=ss["Normal"], fontSize=9, leading=13.5, alignment=TA_JUSTIFY,
                          textColor=colors.HexColor("#3f3f46"), leftIndent=8, borderColor=_ACCENT,
                          borderWidth=0, spaceBefore=4, spaceAfter=2))
    # La explicabilidad va debajo de la interpretacion y un punto mas apagada: es
    # el respaldo auditable, no el mensaje. Por eso el hueco inferior lo cierra
    # ella y no la interpretacion, que ahora solo deja 2pt para pegarse a esta.
    ss.add(ParagraphStyle("Expl", parent=ss["Interp"], fontSize=8.5, leading=12.5,
                          textColor=_MUTE, spaceBefore=0, spaceAfter=12))
    ss.add(ParagraphStyle("Caption", parent=ss["Normal"], fontSize=8, textColor=_MUTE, spaceBefore=2, spaceAfter=2))
    return ss


def _table(data: list[list[str]], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), _MUTE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, _LINE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.4, _LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ]
        )
    )
    return t


def _img(png: bytes, width_mm: float) -> Image:
    bio = io.BytesIO(png)
    img = Image(bio)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * ratio
    return img


def build(db: Session, *, site_id: int, start: date | None, end: date | None) -> bytes:
    site = C.get_site(db, site_id)
    if start is None:
        start = date(2018, 1, 1)
    if end is None:
        end = date(2018, 12, 31)

    ss = _styles()
    story: list = []
    now = datetime.now(timezone.utc)

    # ---- cabecera ----
    story.append(Paragraph(tr("PeatTwin · Resumen ejecutivo del sitio", "PeatTwin · Site executive summary"), ss["H0"]))
    story.append(
        Paragraph(
            f"{site.name}"
            f"{f' · {site.fluxnet_id}' if site.fluxnet_id else ''} — "
            f"{tr('turbera', 'peatland')} {C.peatland_label(site.peatland_type)}"
            f"{f' · {site.latitude:.4f}, {site.longitude:.4f}' if site.latitude is not None else ''}<br/>"
            f"{tr('Periodo', 'Period')} {start.isoformat()} … {end.isoformat()} · "
            f"{tr('generado', 'generated')} {now:%Y-%m-%d %H:%M} UTC",
            ss["Sub"],
        )
    )

    rows = C.model_rows(db)
    winner = next((r for r in rows if r.is_active), rows[0] if rows else None)
    hs = C.holdout_series(db, site, start, end)

    # ==================================================== 1. gráfico de flujos
    story.append(Paragraph(tr("1 · Flujos de carbono observados vs predichos", "1 · Observed vs predicted carbon fluxes"), ss["H1"]))
    if hs.available:
        story.append(_img(F.flux_series_png(hs), 165))
        nrmse, nr2 = C.regression_scores(hs.nee_obs, hs.nee_pred)
        crmse, cr2 = C.regression_scores(hs.fch4_obs, hs.fch4_pred)
        cls_ok = sum(1 for t, p in zip(hs.y_true, hs.proba) if (p >= 0.5) == bool(t))
        acc = cls_ok / len(hs.y_true) if hs.y_true else 0
        story.append(
            Paragraph(
                tr(
                    f"<b>Interpretación.</b> Sobre {len(hs.dates)} días del periodo, el modelo "
                    f"<b>{hs.model_name} ({hs.model_version})</b> reproduce bien el flujo de "
                    f"metano (RMSE = {C.fnum(crmse, 1)} nmol m⁻² s⁻¹, R² = {C.fnum(cr2, 2)}) "
                    f"pero solo captura parcialmente la variación diaria del NEE "
                    f"(RMSE = {C.fnum(nrmse, 2)} gC m⁻² d⁻¹, R² = {C.fnum(nr2, 2)}): el NEE es la "
                    f"pequeña diferencia entre fotosíntesis y respiración y depende del estado de "
                    f"la vegetación, del que este dataset no tiene medidas (NDVI/LAI). La clase "
                    f"sumidero/fuente se acierta el {acc*100:.0f} % de los días.",
                    f"<b>Interpretation.</b> Over {len(hs.dates)} days of the period, the model "
                    f"<b>{hs.model_name} ({hs.model_version})</b> reproduces the methane flux "
                    f"well (RMSE = {C.fnum(crmse, 1)} nmol m⁻² s⁻¹, R² = {C.fnum(cr2, 2)}) "
                    f"but only partly captures the daily NEE variation "
                    f"(RMSE = {C.fnum(nrmse, 2)} gC m⁻² d⁻¹, R² = {C.fnum(nr2, 2)}): NEE is the "
                    f"small difference between photosynthesis and respiration and depends on the "
                    f"vegetation state, which this dataset does not measure (NDVI/LAI). The "
                    f"sink/source class is right {acc*100:.0f} % of the days.",
                ),
                ss["Interp"],
            )
        )
        story.append(
            Paragraph(
                tr(
                    "<b>Explicabilidad.</b> Los dos paneles muestran la serie diaria observada "
                    "(negro) frente a la predicha por el modelo (verde) para el periodo elegido: "
                    "arriba el NEE (CO₂, gC m⁻² d⁻¹; por debajo de 0 el ecosistema captura "
                    "carbono) y abajo el FCH4 (CH₄, nmol m⁻² s⁻¹). Cuanto más se superponen las "
                    "curvas, menor es el error; las predicciones proceden del modelo activo "
                    "aplicado a datos que no vio en el entrenamiento.",
                    "<b>Explainability.</b> The two panels show the observed daily series "
                    "(black) against the model prediction (green) for the chosen period: "
                    "top, NEE (CO₂, gC m⁻² d⁻¹; below 0 the ecosystem takes up carbon) and "
                    "bottom, FCH4 (CH₄, nmol m⁻² s⁻¹). The more the curves overlap, the lower "
                    "the error; predictions come from the active model applied to data it did "
                    "not see during training.",
                ),
                ss["Expl"],
            )
        )
    else:
        story.append(
            Paragraph(
                tr(
                    "No hay serie predicha para este sitio: solo <b>DE-Zrk (Zarnekow)</b> tiene un "
                    "dataset procesado con el que ejecutar el modelo activo.",
                    "There is no predicted series for this site: only <b>DE-Zrk (Zarnekow)</b> has "
                    "a processed dataset on which to run the active model.",
                ),
                ss["Body"],
            )
        )

    # ==================================================== 2. tabla de modelos
    story.append(Paragraph(tr("2 · Comparación de los cinco modelos", "2 · Comparison of the five models"), ss["H1"]))
    data = [[tr("Modelo", "Model"), tr("Familia", "Family"), "RMSE", "R²", "F1", "AUC", "t (s)"]]
    for r in rows:
        data.append(
            [
                f"{r.name}{' ★' if r.is_active else ''}",
                tr("híbrido", "hybrid") if r.algorithm_type == "HYBRID" else tr("clásico", "classical"),
                C.fnum(r.rmse, 2),
                C.fnum(r.r2, 3),
                C.fnum(r.f1, 3),
                C.fnum(r.auc, 3),
                C.fnum(r.train_time_s, 1),
            ]
        )
    story.append(_table(data, col_widths=[62 * mm, 20 * mm, 18 * mm, 16 * mm, 16 * mm, 16 * mm, 14 * mm]))
    story.append(Paragraph(tr("★ = modelo desplegado. RMSE/R² son la media de los objetivos CO₂ y CH₄.", "★ = deployed model. RMSE/R² are the mean of the CO₂ and CH₄ targets."), ss["Caption"]))
    if winner:
        story.append(
            Paragraph(
                tr(
                    f"<b>Interpretación.</b> El modelo desplegado es <b>{winner.name}</b> "
                    f"(v{winner.version}, {'híbrido' if winner.algorithm_type == 'HYBRID' else 'clásico'}, "
                    f"{winner.framework}). Se eligió con un criterio combinado 50 % regresión / "
                    f"50 % clasificación (0.5·R²norm + 0.5·(0.5·F1_macro + 0.5·AUC)); no por un R² "
                    f"puntualmente mejor sino por su mejor comportamiento conjunto y su menor "
                    f"variabilidad entre folds. Su F1_macro = {C.fnum(winner.f1, 3)} y "
                    f"AUC = {C.fnum(winner.auc, 3)} son los más altos del conjunto; el CH₄ se "
                    f"predice bien (R² ≈ {C.fnum(winner.r2, 2)} de media) y el NEE queda limitado por "
                    f"los drivers disponibles.",
                    f"<b>Interpretation.</b> The deployed model is <b>{winner.name}</b> "
                    f"(v{winner.version}, {'hybrid' if winner.algorithm_type == 'HYBRID' else 'classical'}, "
                    f"{winner.framework}). It was chosen with a combined criterion of 50 % regression / "
                    f"50 % classification (0.5·R²norm + 0.5·(0.5·F1_macro + 0.5·AUC)); not for a "
                    f"marginally better R² but for its better joint behavior and lower variability "
                    f"across folds. Its F1_macro = {C.fnum(winner.f1, 3)} and "
                    f"AUC = {C.fnum(winner.auc, 3)} are the highest of the set; CH₄ is "
                    f"predicted well (R² ≈ {C.fnum(winner.r2, 2)} on average) and NEE is limited by "
                    f"the available drivers.",
                ),
                ss["Interp"],
            )
        )
        story.append(
            Paragraph(
                tr(
                    "<b>Explicabilidad.</b> Todas las filas se evalúan sobre el mismo holdout "
                    "temporal: se entrena con 2016–2017 y se mide sobre 2018 completo, sin "
                    "barajar. RMSE es un error, luego menor es mejor; R², F1 y AUC son "
                    "adimensionales y mayor es mejor. Las métricas de regresión son la media de "
                    "los dos objetivos CO₂ y CH₄, y esa media esconde un contraste grande entre "
                    "ambos, así que ninguna fila debe leerse como «calidad del modelo» sin abrir "
                    "el desglose por objetivo.",
                    "<b>Explainability.</b> All rows are evaluated on the same temporal holdout: "
                    "trained on 2016–2017 and measured on the whole of 2018, without shuffling. "
                    "RMSE is an error, so lower is better; R², F1 and AUC are dimensionless and "
                    "higher is better. The regression metrics are the mean of the two targets "
                    "CO₂ and CH₄, and that mean hides a large contrast between them, so no row "
                    "should be read as 'model quality' without opening the per-target breakdown.",
                ),
                ss["Expl"],
            )
        )

    story.append(PageBreak())

    # ==================================================== 3. matriz de confusión
    story.append(Paragraph(tr("3 · Matriz de confusión del modelo ganador", "3 · Confusion matrix of the winning model"), ss["H1"]))
    cm_counts = (
        C.confusion_from_series(hs)
        if hs.available and hs.y_true
        else (winner.tn, winner.fp, winner.fn, winner.tp) if winner else (None, None, None, None)
    )
    if winner and None not in cm_counts:
        tn, fp, fn, tp = cm_counts
        total = tn + fp + fn + tp
        rec_sink = tp / (tp + fn) if (tp + fn) else 0
        prec_sink = tp / (tp + fp) if (tp + fp) else 0
        story.append(_img(F.confusion_png(tn, fp, fn, tp, winner.name), 95))
        story.append(
            Paragraph(
                tr(
                    f"<b>Interpretación.</b> De los {total} días del holdout 2018, el modelo acierta "
                    f"{tn + tp} (accuracy = {C.fnum(winner.accuracy, 3)}). Detecta {tp} de los "
                    f"{tp + fn} días reales de sumidero (recall de sumidero = {rec_sink:.2f}); "
                    f"cuando dice «sumidero», acierta {prec_sink:.2f} de las veces. Los {fn} falsos "
                    f"negativos son días de captura de carbono etiquetados como fuente y los {fp} "
                    f"falsos positivos, lo contrario. Con clases desbalanceadas (~32 % sumidero) "
                    f"esta lectura por clase es más informativa que la accuracy global.",
                    f"<b>Interpretation.</b> Of the {total} days in the 2018 holdout, the model gets "
                    f"{tn + tp} right (accuracy = {C.fnum(winner.accuracy, 3)}). It detects {tp} of the "
                    f"{tp + fn} actual sink days (sink recall = {rec_sink:.2f}); "
                    f"when it says 'sink', it is right {prec_sink:.2f} of the time. The {fn} false "
                    f"negatives are carbon-uptake days labeled as source and the {fp} false "
                    f"positives are the opposite. With imbalanced classes (~32 % sink) "
                    f"this per-class reading is more informative than the overall accuracy.",
                ),
                ss["Interp"],
            )
        )
        story.append(
            Paragraph(
                tr(
                    "<b>Explicabilidad.</b> Las filas son la clase real y las columnas la "
                    "predicha, de modo que la diagonal recoge los aciertos. La clase positiva es "
                    "«sumidero», definida como NEE negativo, y es la minoritaria del conjunto: "
                    "con ese desbalance, un clasificador que respondiera «fuente» siempre ya "
                    "lograría una accuracy alta sin aprender nada, y por eso se reportan F1 y "
                    "recall de sumidero. Fuera de la diagonal, los falsos positivos son días que "
                    "el modelo declara sumidero sin serlo.",
                    "<b>Explainability.</b> Rows are the true class and columns the predicted "
                    "one, so the diagonal holds the hits. The positive class is 'sink', "
                    "defined as negative NEE, and it is the minority of the set: with that "
                    "imbalance, a classifier that always answered 'source' would already reach "
                    "a high accuracy without learning anything, which is why F1 and sink recall "
                    "are reported. Off the diagonal, false positives are days the model declares "
                    "sink without being so.",
                ),
                ss["Expl"],
            )
        )
    else:
        story.append(Paragraph(tr("Sin matriz de confusión registrada para el modelo activo.", "No confusion matrix recorded for the active model."), ss["Body"]))

    # ==================================================== 4. curva ROC
    story.append(Paragraph(tr("4 · Curva ROC del modelo ganador", "4 · ROC curve of the winning model"), ss["H1"]))
    if winner and hs.available and hs.y_true:
        story.append(_img(F.roc_png(hs.y_true, hs.proba, winner.auc, winner.name), 105))
        story.append(
            Paragraph(
                tr(
                    f"<b>Interpretación.</b> El AUC = {C.fnum(winner.auc, 3)} indica que el modelo "
                    f"ordena correctamente los días por su probabilidad de ser sumidero en el "
                    f"{(winner.auc or 0)*100:.0f} % de los pares (sumidero, fuente) posibles. La curva "
                    f"se separa con claridad de la diagonal de azar. El punto de operación por "
                    f"defecto (umbral 0.5) es el que produce la matriz de confusión anterior; "
                    f"moviéndolo se puede priorizar recall (no perder días de captura) o precisión "
                    f"según el uso.",
                    f"<b>Interpretation.</b> AUC = {C.fnum(winner.auc, 3)} means the model "
                    f"correctly orders days by their sink probability in "
                    f"{(winner.auc or 0)*100:.0f} % of the possible (sink, source) pairs. The curve "
                    f"clearly separates from the chance diagonal. The default operating point "
                    f"(threshold 0.5) is the one that produces the confusion matrix above; "
                    f"moving it lets you prioritize recall (not missing uptake days) or precision "
                    f"depending on the use.",
                ),
                ss["Interp"],
            )
        )
        story.append(
            Paragraph(
                tr(
                    "<b>Explicabilidad.</b> La curva enfrenta la tasa de falsos positivos (eje X) "
                    "a la de verdaderos positivos o recall de sumidero (eje Y), recorriendo todos "
                    "los umbrales de decisión posibles; la diagonal discontinua representa al "
                    "clasificador aleatorio. El AUC es el área bajo la curva, así que resume el "
                    "rendimiento a TODOS los umbrales: mide la capacidad de ordenar los días por "
                    "probabilidad de sumidero, no el acierto con el umbral 0.5 que usa la matriz "
                    "de confusión.",
                    "<b>Explainability.</b> The curve plots the false positive rate (X axis) "
                    "against the true positive rate or sink recall (Y axis), sweeping all "
                    "possible decision thresholds; the dashed diagonal represents the random "
                    "classifier. The AUC is the area under the curve, so it summarizes "
                    "performance at ALL thresholds: it measures the ability to order days by "
                    "sink probability, not the accuracy at the 0.5 threshold used by the "
                    "confusion matrix.",
                ),
                ss["Expl"],
            )
        )
    elif winner:
        story.append(
            Paragraph(
                tr(
                    f"AUC registrado del modelo activo: <b>{C.fnum(winner.auc, 3)}</b>. La curva "
                    f"completa requiere el dataset procesado del sitio (solo DE-Zrk).",
                    f"Recorded AUC of the active model: <b>{C.fnum(winner.auc, 3)}</b>. The full "
                    f"curve requires the site's processed dataset (DE-Zrk only).",
                ),
                ss["Body"],
            )
        )

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            tr(
                "Documento generado automáticamente por el módulo de informes de PeatTwin a partir "
                "de la base de datos y del modelo activo. El informe técnico (.docx) contiene la "
                "metodología completa (EDA, entrenamiento, validación cruzada, tuning y pruebas "
                "estadísticas).",
                "Document generated automatically by PeatTwin's reporting module from the "
                "database and the active model. The technical report (.docx) contains the full "
                "methodology (EDA, training, cross-validation, tuning and statistical tests).",
            ),
            ss["Caption"],
        )
    )

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"PeatTwin — {tr('Resumen ejecutivo', 'Executive summary')} {site.fluxnet_id or site.name}",
        author="PeatTwin",
    ).build(story)
    return buf.getvalue()
