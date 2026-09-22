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
                          borderWidth=0, spaceBefore=4, spaceAfter=10))
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
    story.append(Paragraph("PeatTwin · Resumen ejecutivo del sitio", ss["H0"]))
    story.append(
        Paragraph(
            f"{site.name}"
            f"{f' · {site.fluxnet_id}' if site.fluxnet_id else ''} — "
            f"turbera {C.peatland_label(site.peatland_type)}"
            f"{f' · {site.latitude:.4f}, {site.longitude:.4f}' if site.latitude is not None else ''}<br/>"
            f"Periodo {start.isoformat()} … {end.isoformat()} · "
            f"generado {now:%Y-%m-%d %H:%M} UTC",
            ss["Sub"],
        )
    )

    rows = C.model_rows(db)
    winner = next((r for r in rows if r.is_active), rows[0] if rows else None)
    hs = C.holdout_series(db, site, start, end)

    # ==================================================== 1. gráfico de flujos
    story.append(Paragraph("1 · Flujos de carbono observados vs predichos", ss["H1"]))
    if hs.available:
        story.append(_img(F.flux_series_png(hs), 165))
        nrmse, nr2 = C.regression_scores(hs.nee_obs, hs.nee_pred)
        crmse, cr2 = C.regression_scores(hs.fch4_obs, hs.fch4_pred)
        cls_ok = sum(1 for t, p in zip(hs.y_true, hs.proba) if (p >= 0.5) == bool(t))
        acc = cls_ok / len(hs.y_true) if hs.y_true else 0
        story.append(
            Paragraph(
                f"<b>Interpretación.</b> Sobre {len(hs.dates)} días del periodo, el modelo "
                f"<b>{hs.model_name} ({hs.model_version})</b> reproduce bien el flujo de "
                f"metano (RMSE = {C.fnum(crmse, 1)} nmol m⁻² s⁻¹, R² = {C.fnum(cr2, 2)}) "
                f"pero solo captura parcialmente la variación diaria del NEE "
                f"(RMSE = {C.fnum(nrmse, 2)} gC m⁻² d⁻¹, R² = {C.fnum(nr2, 2)}): el NEE es la "
                f"pequeña diferencia entre fotosíntesis y respiración y depende del estado de "
                f"la vegetación, del que este dataset no tiene medidas (NDVI/LAI). La clase "
                f"sumidero/fuente se acierta el {acc*100:.0f} % de los días.",
                ss["Interp"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No hay serie predicha para este sitio: solo <b>DE-Zrk (Zarnekow)</b> tiene un "
                "dataset procesado con el que ejecutar el modelo activo.",
                ss["Body"],
            )
        )

    # ==================================================== 2. tabla de modelos
    story.append(Paragraph("2 · Comparación de los cinco modelos", ss["H1"]))
    data = [["Modelo", "Familia", "RMSE", "R²", "F1", "AUC", "t (s)"]]
    for r in rows:
        data.append(
            [
                f"{r.name}{' ★' if r.is_active else ''}",
                "híbrido" if r.algorithm_type == "HYBRID" else "clásico",
                C.fnum(r.rmse, 2),
                C.fnum(r.r2, 3),
                C.fnum(r.f1, 3),
                C.fnum(r.auc, 3),
                C.fnum(r.train_time_s, 1),
            ]
        )
    story.append(_table(data, col_widths=[62 * mm, 20 * mm, 18 * mm, 16 * mm, 16 * mm, 16 * mm, 14 * mm]))
    story.append(Paragraph("★ = modelo desplegado. RMSE/R² son la media de los objetivos CO₂ y CH₄.", ss["Caption"]))
    if winner:
        story.append(
            Paragraph(
                f"<b>Interpretación.</b> El modelo desplegado es <b>{winner.name}</b> "
                f"(v{winner.version}, {'híbrido' if winner.algorithm_type == 'HYBRID' else 'clásico'}, "
                f"{winner.framework}). Se eligió con un criterio combinado 50 % regresión / "
                f"50 % clasificación (0.5·R²norm + 0.5·(0.5·F1_macro + 0.5·AUC)); no por un R² "
                f"puntualmente mejor sino por su mejor comportamiento conjunto y su menor "
                f"variabilidad entre folds. Su F1_macro = {C.fnum(winner.f1, 3)} y "
                f"AUC = {C.fnum(winner.auc, 3)} son los más altos del conjunto; el CH₄ se "
                f"predice bien (R² ≈ {C.fnum(winner.r2, 2)} de media) y el NEE queda limitado por "
                f"los drivers disponibles.",
                ss["Interp"],
            )
        )

    story.append(PageBreak())

    # ==================================================== 3. matriz de confusión
    story.append(Paragraph("3 · Matriz de confusión del modelo ganador", ss["H1"]))
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
                f"<b>Interpretación.</b> De los {total} días del holdout 2018, el modelo acierta "
                f"{tn + tp} (accuracy = {C.fnum(winner.accuracy, 3)}). Detecta {tp} de los "
                f"{tp + fn} días reales de sumidero (recall de sumidero = {rec_sink:.2f}); "
                f"cuando dice «sumidero», acierta {prec_sink:.2f} de las veces. Los {fn} falsos "
                f"negativos son días de captura de carbono etiquetados como fuente y los {fp} "
                f"falsos positivos, lo contrario. Con clases desbalanceadas (~32 % sumidero) "
                f"esta lectura por clase es más informativa que la accuracy global.",
                ss["Interp"],
            )
        )
    else:
        story.append(Paragraph("Sin matriz de confusión registrada para el modelo activo.", ss["Body"]))

    # ==================================================== 4. curva ROC
    story.append(Paragraph("4 · Curva ROC del modelo ganador", ss["H1"]))
    if winner and hs.available and hs.y_true:
        story.append(_img(F.roc_png(hs.y_true, hs.proba, winner.auc, winner.name), 105))
        story.append(
            Paragraph(
                f"<b>Interpretación.</b> El AUC = {C.fnum(winner.auc, 3)} indica que el modelo "
                f"ordena correctamente los días por su probabilidad de ser sumidero en el "
                f"{(winner.auc or 0)*100:.0f} % de los pares (sumidero, fuente) posibles. La curva "
                f"se separa con claridad de la diagonal de azar. El punto de operación por "
                f"defecto (umbral 0.5) es el que produce la matriz de confusión anterior; "
                f"moviéndolo se puede priorizar recall (no perder días de captura) o precisión "
                f"según el uso.",
                ss["Interp"],
            )
        )
    elif winner:
        story.append(
            Paragraph(
                f"AUC registrado del modelo activo: <b>{C.fnum(winner.auc, 3)}</b>. La curva "
                f"completa requiere el dataset procesado del sitio (solo DE-Zrk).",
                ss["Body"],
            )
        )

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Documento generado automáticamente por el módulo de informes de PeatTwin a partir "
            "de la base de datos y del modelo activo. El informe técnico (.docx) contiene la "
            "metodología completa (EDA, entrenamiento, validación cruzada, tuning y pruebas "
            "estadísticas).",
            ss["Caption"],
        )
    )

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"PeatTwin — Resumen ejecutivo {site.fluxnet_id or site.name}",
        author="PeatTwin",
    ).build(story)
    return buf.getvalue()
