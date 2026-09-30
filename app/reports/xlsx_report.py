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

from app.core.i18n import tr
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
    ws = wb.create_sheet(_N_LECTURA())
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

    _titulo(tr("PeatTwin - guia de lectura de este libro", "PeatTwin - reading guide for this workbook"))
    _linea(
        tr("Sitio", "Site"), site.name,
        tr(
            "Gemelo digital de turbera. Los datos observados proceden de FLUXNET-CH4 "
            "(producto diario) y las predicciones, de ejecutar el modelo desplegado sobre "
            "esas mismas variables.",
            "Peatland digital twin. Observed data come from FLUXNET-CH4 (daily product) "
            "and predictions from running the deployed model on those same variables.",
        ),
    )
    _linea(
        tr("Modelo desplegado", "Deployed model"), winner.name if winner else "-",
        tr(
            f"Es el modelo que genero la hoja `predictions`. La hoja `{_N_MODELO()}` "
            f"dice como de bien lo hace, y `{_N_FOLD()}`, si ese resultado es estable.",
            f"It is the model that produced the `predictions` sheet. The `{_N_MODELO()}` "
            f"sheet says how well it does, and `{_N_FOLD()}`, whether that result is stable.",
        ),
    )
    fila += 1

    _titulo(tr("Que hay en cada hoja", "What each sheet contains"))
    _cabecera(tr("Hoja", "Sheet"), tr("Contiene", "Contains"), tr("Como leerla", "How to read it"))
    _linea(
        "info", tr("Metadatos del sitio y del modelo", "Site and model metadata"),
        tr(
            "Contexto de la exportacion: fecha UTC de generacion, coordenadas y version del "
            "modelo activo. Sirve para saber a que ejecucion corresponde el resto del libro.",
            "Export context: UTC generation date, coordinates and active model version. "
            "It tells which run the rest of the workbook corresponds to.",
        ),
    )
    _linea(
        "flux_observations", tr("Serie diaria medida", "Measured daily series"),
        tr(
            "Una fila por dia. Son medidas ya procesadas por el equipo del sitio, incluido el "
            "relleno de huecos con red neuronal (ANNOPTLM): no son datos crudos de torre. "
            "Ojo al periodo: el nivel freatico esta 100 % vacio en 2013-2015, por eso el "
            "analisis del proyecto se restringe a 2016-2018.",
            "One row per day. These are measurements already processed by the site team, "
            "including neural-network gap-filling (ANNOPTLM): they are not raw tower data. "
            "Mind the period: the water table is 100 % empty in 2013-2015, which is why the "
            "project's analysis is restricted to 2016-2018.",
        ),
    )
    _linea(
        "predictions", tr("Escenarios simulados y guardados", "Simulated and saved scenarios"),
        tr(
            "NO es el backtest de la serie: son las simulaciones que los usuarios han lanzado "
            "y guardado desde el simulador, cada una con su nivel freatico objetivo. El resto "
            "de drivers en esas filas proceden de la climatologia del dia del anio, no de "
            "medidas reales.",
            "It is NOT the series backtest: these are the simulations users have run and "
            "saved from the simulator, each with its target water table. The remaining "
            "drivers in those rows come from the day-of-year climatology, not from real "
            "measurements.",
        ),
    )
    _linea(
        _N_MODELO(), tr("Resultado en el holdout de 2018", "Result on the 2018 holdout"),
        tr(
            "Una fila por modelo, evaluada sobre el mismo corte temporal (entrena 2016-2017, "
            "test 2018, sin barajar). Las metricas de regresion son la MEDIA de los dos "
            "objetivos CO2 y CH4, y esa media esconde un contraste grande: el CH4 se predice "
            "bien (R2 ~ 0.8) y el CO2 mal (R2 ~ 0.2). No leer una fila como «calidad del "
            "modelo» sin mirar el desglose por objetivo.",
            "One row per model, evaluated on the same temporal cut (trains 2016-2017, tests "
            "2018, no shuffling). The regression metrics are the MEAN of the two targets CO2 "
            "and CH4, and that mean hides a large contrast: CH4 is predicted well (R2 ~ 0.8) "
            "and CO2 poorly (R2 ~ 0.2). Do not read a row as 'model quality' without looking "
            "at the per-target breakdown.",
        ),
    )
    _linea(
        _N_FOLD(), tr("Validacion cruzada temporal", "Temporal cross-validation"),
        tr(
            "La misma comparacion repetida sobre ventanas crecientes (TimeSeriesSplit). "
            "fold = -1 es el holdout unico de la hoja anterior. La dispersion entre folds mide "
            "ESTABILIDAD: un modelo con media alta y desviacion alta es peor apuesta que uno "
            "algo peor pero constante. Los primeros folds entrenan con menos de medio ciclo "
            "estacional y dan R2 negativos; es un hallazgo sobre suficiencia de datos, no un "
            "fallo de calculo.",
            "The same comparison repeated over growing windows (TimeSeriesSplit). "
            "fold = -1 is the single holdout of the previous sheet. The spread across folds "
            "measures STABILITY: a model with a high mean and high deviation is a worse bet "
            "than one slightly worse but steady. The first folds train on less than half a "
            "seasonal cycle and give negative R2; it is a finding about data sufficiency, not "
            "a calculation failure.",
        ),
    )
    fila += 1

    _titulo(tr("Unidades y convenios de signo", "Units and sign conventions"))
    _cabecera(tr("Columna", "Column"), tr("Unidad", "Unit"), tr("Que significa", "What it means"))
    _linea(
        "nee_co2 / " + tr("co2_predicho", "co2_predicted"), "gC m-2 d-1",
        tr(
            "Intercambio neto de CO2. SIGNO: negativo = la turbera captura carbono "
            "(sumidero); positivo = lo emite (fuente). Es la diferencia entre dos flujos "
            "grandes (fotosintesis y respiracion), de ahi que sea dificil de predecir.",
            "Net CO2 exchange. SIGN: negative = the peatland takes up carbon (sink); "
            "positive = it emits it (source). It is the difference between two large fluxes "
            "(photosynthesis and respiration), hence hard to predict.",
        ),
    )
    _linea(
        "fch4 / " + tr("ch4_predicho", "ch4_predicted"), "nmol CH4 m-2 s-1",
        tr(
            "Flujo de metano. Siempre positivo: la turbera emite metano todos los dias. "
            "Unidad distinta de la del CO2: para sumar ambos gases hay que convertir a "
            "CO2-equivalente y declarar el horizonte de GWP que se usa.",
            "Methane flux. Always positive: the peatland emits methane every day. A different "
            "unit from CO2: to add both gases you must convert to CO2-equivalent and state "
            "the GWP horizon used.",
        ),
    )
    _linea(
        "water_table_depth_cm", "cm",
        tr(
            "Nivel freatico. Positivo = lamina de agua POR ENCIMA de la superficie. Es la "
            "variable que mueve el simulador de escenarios de restauracion.",
            "Water table. Positive = water layer ABOVE the surface. It is the variable the "
            "restoration scenario simulator moves.",
        ),
    )
    _linea(
        "soil_temp_c / air_temp_c", tr("grados C", "degrees C"),
        tr(
            "Temperatura de suelo y de aire. La de suelo es la media del perfil TS_1..TS_5, "
            "que se promedian por su colinealidad (r > 0.95).",
            "Soil and air temperature. Soil temperature is the mean of the TS_1..TS_5 "
            "profile, averaged because of their collinearity (r > 0.95).",
        ),
    )
    _linea(
        "soil_water_content", "-",
        tr(
            "Columna vacia en este sitio: Zarnekow no midio humedad del suelo. Es una "
            "limitacion registrada del dataset, no un error de la exportacion.",
            "Empty column at this site: Zarnekow did not measure soil moisture. It is a "
            "recorded limitation of the dataset, not an export error.",
        ),
    )
    _linea(
        "precipitation_mm", "mm d-1",
        tr("Precipitacion diaria acumulada.", "Accumulated daily precipitation."),
    )
    _linea(
        "carbon_balance_class", tr("sumidero / fuente", "sink / source"),
        tr(
            "Clase derivada del SIGNO del NEE observado: sumidero si NEE < 0. Esta "
            "desbalanceada (~32 % sumidero), asi que la accuracy de un clasificador hay que "
            "compararla contra el ~68 % que saca uno que responda «fuente» siempre, no contra "
            "el 50 %.",
            "Class derived from the SIGN of the observed NEE: sink if NEE < 0. It is "
            "imbalanced (~32 % sink), so a classifier's accuracy must be compared against the "
            "~68 % that one always answering 'source' gets, not against 50 %.",
        ),
    )
    _linea(
        "quality_flag", "-",
        tr(
            "Procedencia del dato segun FLUXNET: distingue medido de rellenado por el "
            "gap-filling del sitio.",
            "Data provenance according to FLUXNET: it distinguishes measured from filled by "
            "the site's gap-filling.",
        ),
    )
    fila += 1

    _titulo(tr("Metricas", "Metrics"))
    _cabecera(tr("Metrica", "Metric"), tr("Rango", "Range"), tr("Que mide", "What it measures"))
    _linea("rmse / mae", tr("0 -> inf, menor mejor", "0 -> inf, lower is better"),
           tr("Error en la unidad del objetivo. RMSE penaliza mas los errores grandes.",
              "Error in the target's unit. RMSE penalizes large errors more."))
    _linea("r2", tr("-inf -> 1, mayor mejor", "-inf -> 1, higher is better"),
           tr("Fraccion de varianza explicada. Un R2 de 0 equivale a predecir siempre la "
              "media; negativo significa que el modelo es peor que esa media.",
              "Fraction of variance explained. An R2 of 0 equals always predicting the "
              "mean; negative means the model is worse than that mean."))
    _linea("nse", tr("identico a r2", "identical to r2"),
           tr("Nash-Sutcliffe. Coincide con R2 por definicion cuando se calcula asi; se "
              "reporta porque es el estandar en hidrologia.",
              "Nash-Sutcliffe. Equals R2 by definition when computed this way; it is "
              "reported because it is the standard in hydrology."))
    _linea("kge", tr("-inf -> 1, mayor mejor", "-inf -> 1, higher is better"),
           tr("Kling-Gupta: descompone el error en correlacion, sesgo y variabilidad. NO esta "
              "definido para el NEE porque su media es ~ 0 y el termino de sesgo se dispara.",
              "Kling-Gupta: decomposes the error into correlation, bias and variability. It is "
              "NOT defined for NEE because its mean is ~ 0 and the bias term blows up."))
    _linea("accuracy", "0 -> 1",
           tr("Aciertos sobre el total. Enganosa con clases desbalanceadas: leer f1 y auc.",
              "Hits over the total. Misleading with imbalanced classes: read f1 and auc."))
    _linea("f1", tr("0 -> 1, mayor mejor", "0 -> 1, higher is better"),
           tr("Media armonica de precision y recall. Es la metrica de clasificacion que usa "
              "el criterio de seleccion, en su version macro.",
              "Harmonic mean of precision and recall. It is the classification metric used by "
              "the selection criterion, in its macro version."))
    _linea("auc / auc_roc", tr("0.5 = azar, 1 = perfecto", "0.5 = chance, 1 = perfect"),
           tr("Capacidad de ORDENAR los dias por probabilidad de sumidero, a todos los "
              "umbrales. No depende del umbral 0.5 que usa la matriz de confusion.",
              "Ability to RANK days by sink probability, at all thresholds. It does not "
              "depend on the 0.5 threshold used by the confusion matrix."))
    _linea("tn / fp / fn / tp", tr("conteos", "counts"),
           tr("Matriz de confusion con sumidero como clase positiva. Para restauracion los "
              "fp son el error caro: dias declarados sumidero que no lo eran, es decir "
              "beneficio sobreestimado.",
              "Confusion matrix with sink as the positive class. For restoration, fp are the "
              "costly error: days declared sink that were not, i.e. overestimated benefit."))
    _linea("fold", "-1, 0, 1, ...",
           tr("-1 = holdout unico de 2018. El resto son los cortes de la validacion cruzada "
              "temporal, en orden cronologico creciente.",
              "-1 = single 2018 holdout. The rest are the temporal cross-validation cuts, in "
              "increasing chronological order."))
    fila += 1

    _titulo(tr("Advertencia", "Warning"))
    _linea(
        "", "",
        tr(
            "Un solo sitio (DE-Zrk) y tres anios: los resultados describen ESTA turbera en "
            "ESTE periodo y no se pueden extrapolar a otras sin reentrenar y volver a validar. "
            "Las curvas de respuesta del simulador mueven el nivel freatico dejando el resto "
            "de variables congeladas en su climatologia, asi que son sensibilidad del modelo, "
            "no una prediccion de lo que ocurriria al rehumedecer de verdad.",
            "A single site (DE-Zrk) and three years: the results describe THIS peatland in "
            "THIS period and cannot be extrapolated to others without retraining and "
            "revalidating. The simulator's response curves move the water table while leaving "
            "the other variables frozen at their climatology, so they are model sensitivity, "
            "not a prediction of what would happen on real rewetting.",
        ),
    )

    ws.freeze_panes = "A2"


def _N_LECTURA() -> str:
    return tr("lectura", "guide")


def _N_MODELO() -> str:
    return tr("metricas_por_modelo", "metrics_by_model")


def _N_FOLD() -> str:
    return tr("metricas_por_fold", "metrics_by_fold")


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
        [tr("clave", "key"), tr("valor", "value")],
        [
            [tr("generado_utc", "generated_utc"), datetime.now(timezone.utc).replace(tzinfo=None)],
            [tr("sitio_id", "site_id"), site.id],
            [tr("sitio_nombre", "site_name"), site.name],
            ["fluxnet_id", site.fluxnet_id or ""],
            ["peatland_type", str(getattr(site.peatland_type, "value", site.peatland_type) or "")],
            [tr("latitud", "latitude"), site.latitude],
            [tr("longitud", "longitude"), site.longitude],
            [tr("modelo_activo", "active_model"), winner.name if winner else ""],
            [tr("modelo_activo_version", "active_model_version"), winner.version if winner else ""],
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
        ["id", tr("escenario", "scenario"), tr("modelo", "model"), "timestamp", tr("co2_predicho", "co2_predicted"), tr("ch4_predicho", "ch4_predicted"), tr("clase_predicha", "predicted_class")],
        [
            [p["id"], p["escenario"], p["modelo"], _cell(p["timestamp"]),
             p["co2_predicted"], p["ch4_predicted"],
             str(getattr(p["predicted_class"], "value", p["predicted_class"]) or "")]
            for p in preds
        ],
    )

    # ---- metricas por modelo (holdout, fold = -1) ----
    _sheet(
        wb, _N_MODELO(),
        [tr("modelo", "model"), tr("familia", "family"), "framework", "version", tr("activo", "active"), "rmse", "mae", "r2", "nse",
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
        wb, _N_FOLD(),
        [tr("modelo", "model"), tr("tarea", "task"), "version", "fold", "rmse", "mae", "r2", "nse", "kge", "accuracy",
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
