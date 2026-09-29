"""Informe Excel (.xlsx): exportacion cruda de flux_observations + predictions +
tabla completa de metricas por modelo y por fold.

A diferencia del PDF y del Word, aqui no hay figuras ni parrafos: son datos para
seguir trabajando. Aun asi vale la regla del proyecto de que ninguna cifra viaje
sin explicacion, asi que el libro abre con la hoja `lectura`, que dice que hay
en cada hoja, en que unidades y con que salvedades. Sin ella un `nee_co2` de
-1.8 no se puede interpretar: ni la unidad ni el signo son evidentes.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sqlalchemy.orm import Session

from app.reports import common as C

_HEADER_FILL = PatternFill("solid", fgColor="ECFDF5")
_HEADER_FONT = Font(bold=True, color="047857")


def _sheet(wb: Workbook, name: str, headers: list[str], rows: list[list]):
    ws = wb.create_sheet(name[:31])
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="left")
    for r in rows:
        ws.append(r)
    ws.freeze_panes = "A2"
    # ancho de columna aproximado
    for c, h in enumerate(headers, start=1):
        maxlen = max([len(str(h))] + [len(str(r[c - 1])) for r in rows[:200] if c - 1 < len(r)] or [0])
        ws.column_dimensions[get_column_letter(c)].width = min(max(maxlen + 2, 10), 40)
    return ws


_TITLE_FONT = Font(bold=True, size=12, color="047857")


def _lectura_sheet(wb: Workbook, site, winner) -> None:
    """Primera hoja del libro: guia de lectura y diccionario de unidades.

    Va la primera a proposito, porque es la que Excel abre al cargar el fichero.
    """
    ws = wb.create_sheet("lectura")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 96

    fila = 1

    def _titulo(texto: str) -> None:
        nonlocal fila
        c = ws.cell(row=fila, column=1, value=texto)
        c.font = _TITLE_FONT
        fila += 1

    def _linea(a: str = "", b: str = "", c: str = "") -> None:
        nonlocal fila
        ws.cell(row=fila, column=1, value=a)
        ws.cell(row=fila, column=2, value=b)
        celda = ws.cell(row=fila, column=3, value=c)
        celda.alignment = Alignment(wrap_text=True, vertical="top")
        fila += 1

    def _cabecera(a: str, b: str, c: str) -> None:
        nonlocal fila
        for col, val in ((1, a), (2, b), (3, c)):
            celda = ws.cell(row=fila, column=col, value=val)
            celda.fill = _HEADER_FILL
            celda.font = _HEADER_FONT
        fila += 1

    _titulo("PeatTwin - guia de lectura de este libro")
    _linea(
        "Sitio", site.name,
        "Gemelo digital de turbera. Los datos observados proceden de FLUXNET-CH4 "
        "(producto diario) y las predicciones, de ejecutar el modelo desplegado sobre "
        "esas mismas variables.",
    )
    _linea(
        "Modelo desplegado", winner.name if winner else "-",
        "Es el modelo que genero la hoja `predictions`. La hoja `metricas_por_modelo` "
        "dice como de bien lo hace, y `metricas_por_fold`, si ese resultado es estable.",
    )
    fila += 1

    _titulo("Que hay en cada hoja")
    _cabecera("Hoja", "Contiene", "Como leerla")
    _linea(
        "info", "Metadatos del sitio y del modelo",
        "Contexto de la exportacion: fecha UTC de generacion, coordenadas y version del "
        "modelo activo. Sirve para saber a que ejecucion corresponde el resto del libro.",
    )
    _linea(
        "flux_observations", "Serie diaria medida",
        "Una fila por dia. Son medidas ya procesadas por el equipo del sitio, incluido el "
        "relleno de huecos con red neuronal (ANNOPTLM): no son datos crudos de torre. "
        "Ojo al periodo: el nivel freatico esta 100 % vacio en 2013-2015, por eso el "
        "analisis del proyecto se restringe a 2016-2018.",
    )
    _linea(
        "predictions", "Escenarios simulados y guardados",
        "NO es el backtest de la serie: son las simulaciones que los usuarios han lanzado "
        "y guardado desde el simulador, cada una con su nivel freatico objetivo. El resto "
        "de drivers en esas filas proceden de la climatologia del dia del anio, no de "
        "medidas reales.",
    )
    _linea(
        "metricas_por_modelo", "Resultado en el holdout de 2018",
        "Una fila por modelo, evaluada sobre el mismo corte temporal (entrena 2016-2017, "
        "test 2018, sin barajar). Las metricas de regresion son la MEDIA de los dos "
        "objetivos CO2 y CH4, y esa media esconde un contraste grande: el CH4 se predice "
        "bien (R2 ~ 0.8) y el CO2 mal (R2 ~ 0.2). No leer una fila como «calidad del "
        "modelo» sin mirar el desglose por objetivo.",
    )
    _linea(
        "metricas_por_fold", "Validacion cruzada temporal",
        "La misma comparacion repetida sobre ventanas crecientes (TimeSeriesSplit). "
        "fold = -1 es el holdout unico de la hoja anterior. La dispersion entre folds mide "
        "ESTABILIDAD: un modelo con media alta y desviacion alta es peor apuesta que uno "
        "algo peor pero constante. Los primeros folds entrenan con menos de medio ciclo "
        "estacional y dan R2 negativos; es un hallazgo sobre suficiencia de datos, no un "
        "fallo de calculo.",
    )
    fila += 1

    _titulo("Unidades y convenios de signo")
    _cabecera("Columna", "Unidad", "Que significa")
    _linea(
        "nee_co2 / co2_predicho", "gC m-2 d-1",
        "Intercambio neto de CO2. SIGNO: negativo = la turbera captura carbono "
        "(sumidero); positivo = lo emite (fuente). Es la diferencia entre dos flujos "
        "grandes (fotosintesis y respiracion), de ahi que sea dificil de predecir.",
    )
    _linea(
        "fch4 / ch4_predicho", "nmol CH4 m-2 s-1",
        "Flujo de metano. Siempre positivo: la turbera emite metano todos los dias. "
        "Unidad distinta de la del CO2: para sumar ambos gases hay que convertir a "
        "CO2-equivalente y declarar el horizonte de GWP que se usa.",
    )
    _linea(
        "water_table_depth_cm", "cm",
        "Nivel freatico. Positivo = lamina de agua POR ENCIMA de la superficie. Es la "
        "variable que mueve el simulador de escenarios de restauracion.",
    )
    _linea(
        "soil_temp_c / air_temp_c", "grados C",
        "Temperatura de suelo y de aire. La de suelo es la media del perfil TS_1..TS_5, "
        "que se promedian por su colinealidad (r > 0.95).",
    )
    _linea(
        "soil_water_content", "-",
        "Columna vacia en este sitio: Zarnekow no midio humedad del suelo. Es una "
        "limitacion registrada del dataset, no un error de la exportacion.",
    )
    _linea(
        "precipitation_mm", "mm d-1",
        "Precipitacion diaria acumulada.",
    )
    _linea(
        "carbon_balance_class", "sumidero / fuente",
        "Clase derivada del SIGNO del NEE observado: sumidero si NEE < 0. Esta "
        "desbalanceada (~32 % sumidero), asi que la accuracy de un clasificador hay que "
        "compararla contra el ~68 % que saca uno que responda «fuente» siempre, no contra "
        "el 50 %.",
    )
    _linea(
        "quality_flag", "-",
        "Procedencia del dato segun FLUXNET: distingue medido de rellenado por el "
        "gap-filling del sitio.",
    )
    fila += 1

    _titulo("Metricas")
    _cabecera("Metrica", "Rango", "Que mide")
    _linea("rmse / mae", "0 -> inf, menor mejor",
           "Error en la unidad del objetivo. RMSE penaliza mas los errores grandes.")
    _linea("r2", "-inf -> 1, mayor mejor",
           "Fraccion de varianza explicada. Un R2 de 0 equivale a predecir siempre la "
           "media; negativo significa que el modelo es peor que esa media.")
    _linea("nse", "identico a r2",
           "Nash-Sutcliffe. Coincide con R2 por definicion cuando se calcula asi; se "
           "reporta porque es el estandar en hidrologia.")
    _linea("kge", "-inf -> 1, mayor mejor",
           "Kling-Gupta: descompone el error en correlacion, sesgo y variabilidad. NO esta "
           "definido para el NEE porque su media es ~ 0 y el termino de sesgo se dispara.")
    _linea("accuracy", "0 -> 1",
           "Aciertos sobre el total. Enganosa con clases desbalanceadas: leer f1 y auc.")
    _linea("f1", "0 -> 1, mayor mejor",
           "Media armonica de precision y recall. Es la metrica de clasificacion que usa "
           "el criterio de seleccion, en su version macro.")
    _linea("auc / auc_roc", "0.5 = azar, 1 = perfecto",
           "Capacidad de ORDENAR los dias por probabilidad de sumidero, a todos los "
           "umbrales. No depende del umbral 0.5 que usa la matriz de confusion.")
    _linea("tn / fp / fn / tp", "conteos",
           "Matriz de confusion con sumidero como clase positiva. Para restauracion los "
           "fp son el error caro: dias declarados sumidero que no lo eran, es decir "
           "beneficio sobreestimado.")
    _linea("fold", "-1, 0, 1, ...",
           "-1 = holdout unico de 2018. El resto son los cortes de la validacion cruzada "
           "temporal, en orden cronologico creciente.")
    fila += 1

    _titulo("Advertencia")
    _linea(
        "", "",
        "Un solo sitio (DE-Zrk) y tres anios: los resultados describen ESTA turbera en "
        "ESTE periodo y no se pueden extrapolar a otras sin reentrenar y volver a validar. "
        "Las curvas de respuesta del simulador mueven el nivel freatico dejando el resto "
        "de variables congeladas en su climatologia, asi que son sensibilidad del modelo, "
        "no una prediccion de lo que ocurriria al rehumedecer de verdad.",
    )

    ws.freeze_panes = "A2"


def _cell(v):
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    return v


def build(db: Session, *, site_id: int) -> bytes:
    site = C.get_site(db, site_id)
    wb = Workbook()
    wb.remove(wb.active)  # quita la hoja por defecto

    # ---- lectura (primera hoja: es la que Excel abre) ----
    rows_active = C.model_rows(db)
    winner = next((r for r in rows_active if r.is_active), None)
    _lectura_sheet(wb, site, winner)

    # ---- info ----
    _sheet(
        wb, "info",
        ["clave", "valor"],
        [
            ["generado_utc", datetime.now(timezone.utc).replace(tzinfo=None)],
            ["sitio_id", site.id],
            ["sitio_nombre", site.name],
            ["fluxnet_id", site.fluxnet_id or ""],
            ["peatland_type", str(getattr(site.peatland_type, "value", site.peatland_type) or "")],
            ["latitud", site.latitude],
            ["longitud", site.longitude],
            ["modelo_activo", winner.name if winner else ""],
            ["modelo_activo_version", winner.version if winner else ""],
        ],
    )

    # ---- flux_observations ----
    obs = C.flux_rows(db, site_id, None, None)
    _sheet(
        wb, "flux_observations",
        [
            "id", "timestamp", "nee_co2", "fch4", "water_table_depth_cm", "soil_temp_c",
            "soil_water_content", "air_temp_c", "precipitation_mm", "carbon_balance_class",
            "quality_flag",
        ],
        [
            [
                o.id, _cell(o.timestamp), o.nee_co2, o.fch4, o.water_table_depth, o.soil_temp,
                o.soil_water_content, o.air_temp, o.precipitation,
                str(getattr(o.carbon_balance_class, "value", o.carbon_balance_class) or ""),
                o.quality_flag,
            ]
            for o in obs
        ],
    )

    # ---- predictions ----
    preds = C.prediction_rows(db, site_id)
    _sheet(
        wb, "predictions",
        ["id", "escenario", "modelo", "timestamp", "co2_predicho", "ch4_predicho", "clase_predicha"],
        [
            [p["id"], p["escenario"], p["modelo"], _cell(p["timestamp"]),
             p["co2_predicted"], p["ch4_predicted"],
             str(getattr(p["predicted_class"], "value", p["predicted_class"]) or "")]
            for p in preds
        ],
    )

    # ---- metricas por modelo (holdout, fold = -1) ----
    _sheet(
        wb, "metricas_por_modelo",
        ["modelo", "familia", "framework", "version", "activo", "rmse", "mae", "r2", "nse",
         "kge", "accuracy", "precision", "recall", "f1", "auc", "tn", "fp", "fn", "tp",
         "training_time_s"],
        [
            [
                r.name, r.algorithm_type, r.framework, r.version, r.is_active,
                r.rmse, r.mae, r.r2, r.nse, r.kge, r.accuracy, r.precision, r.recall, r.f1,
                r.auc, r.tn, r.fp, r.fn, r.tp, r.train_time_s,
            ]
            for r in rows_active
        ],
    )

    # ---- metricas por modelo y por fold (todas las filas de ml_model_metrics) ----
    fm = C.fold_metric_rows(db)
    _sheet(
        wb, "metricas_por_fold",
        ["modelo", "tarea", "version", "fold", "rmse", "mae", "r2", "nse", "kge", "accuracy",
         "precision", "recall", "f1", "auc_roc", "training_time_seconds"],
        [
            [
                m["modelo"], m["tarea"], m["version"], m["fold"], m["rmse"], m["mae"], m["r2"],
                m["nse"], m["kge"], m["accuracy"], m["precision"], m["recall"], m["f1"],
                m["auc_roc"], m["training_time_seconds"],
            ]
            for m in fm
        ],
    )

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
