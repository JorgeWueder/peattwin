"""Motor IA — analitica del pipeline de ML. Vista exclusiva del rol `admin`.

Diez subsecciones en el orden del proyecto de referencia, adaptadas a que aqui el
problema es REGRESION doble (NEE y FCH4) con clasificacion auxiliar del balance
de carbono, no clasificacion de fallas.

Salvo la subseccion de reentrenamiento, todo lee artefactos que el pipeline ya
dejo en disco: abrir una pestana no reentrena nada.

La navegacion interna usa `st.segmented_control` y no `st.tabs` a proposito:
Streamlit ejecuta el cuerpo de TODAS las pestanas en cada rerun, lo que cargaria
los diez bloques de artefactos cada vez.

Todos los imports estan a nivel de modulo. Ninguno dentro de una funcion: es lo
que evita el `UnboundLocalError` por sombra de un nombre global.

Idioma: la seleccion de subseccion guarda un identificador estable (no la
etiqueta), asi que cambiar es/en no la pierde.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import streamlit as st

from app.analytics import artifacts as A
from app.analytics import predict_panel as P
from app.analytics import retrain as R
from ui import auth, charts
from ui import format as F
from ui import interpret as I
from ui.i18n import tr

# Prefijo `mia_` en las claves de sesion: identificables y barridas por el logout.
_CLAVE_SECCION = "mia_seccion"
_CLAVE_RESULTADO = "mia_resultado_reentrenamiento"

SECCIONES = [
    "eda", "entrenamiento", "comparativa", "prediccion", "mejor", "reentrenar",
    "logs", "cv", "hiperparametros", "estadisticas",
]


def _etiqueta_seccion(clave: str) -> str:
    return {
        "eda": "🔍 EDA",
        "entrenamiento": tr("⚙️ Entrenamiento", "⚙️ Training"),
        "comparativa": tr("📊 Comparativa", "📊 Comparison"),
        "prediccion": tr("🔮 Prediccion", "🔮 Prediction"),
        "mejor": tr("🏆 Mejor modelo", "🏆 Best model"),
        "reentrenar": tr("🔄 Reentrenar", "🔄 Retrain"),
        "logs": "📋 Logs",
        "cv": tr("🔁 Validacion cruzada", "🔁 Cross-validation"),
        "hiperparametros": tr("🎛️ Hiperparametros", "🎛️ Hyperparameters"),
        "estadisticas": tr("🧪 Pruebas estadisticas", "🧪 Statistical tests"),
    }[clave]


def _clase(nombre: str) -> str:
    return {"sumidero": tr("sumidero", "sink"), "fuente": tr("fuente", "source")}.get(
        nombre, nombre
    )


_CMD_EDA = "python -m ml.eda.run_eda"
_CMD_TRAIN = "python -m ml.training.run_training --save-all"
_CMD_CV = "python -m ml.training.cross_validation --folds 5"
_CMD_TUNING = "python -m ml.training.tuning"
_CMD_STATS = "python -m ml.training.statistical_tests"

_NOMBRE_CORTO = {
    "Random Forest": "Random Forest",
    "Gradient Boosting (XGBoost)": "XGBoost",
    "SVR / SVM": "SVR / SVM",
    "CNN-LSTM": "CNN-LSTM",
    "Stacking Ensemble": "Stacking",
    "Stacking Ensemble (tuned)": "Stacking (tuned)",
    "Random Forest (tuned)": "Random Forest (tuned)",
    "Gradient Boosting (XGBoost) (tuned)": "XGBoost (tuned)",
}


# ============================================================ 1 · EDA
def _eda() -> None:
    I.titulo(tr("Exploracion de datos (EDA)", "Exploratory data analysis (EDA)"),
             ayuda=tr("DE-Zrk (Zarnekow), serie diaria 2016-2018 · 1096 dias.",
                      "DE-Zrk (Zarnekow), daily series 2016-2018 · 1096 days."))

    desc = A.descriptivos()
    if desc.empty:
        I.falta_artefacto(tr("las tablas del EDA", "the EDA tables"), _CMD_EDA)
        return

    I.tabla(
        desc,
        explicabilidad=tr(
            "Estadisticos de posicion, dispersion y forma de las 8 variables clave "
            "sobre los 1096 dias de la serie diaria de DE-Zrk (2016-2018). `n` es el "
            "numero de dias con dato; media y mediana van en la unidad de cada "
            "variable (NEE en gC m⁻² d⁻¹, FCH₄ en nmol m⁻² s⁻¹, TS en °C); la "
            "asimetria y la curtosis de exceso valen 0 en una normal.",
            "Location, dispersion and shape statistics of the 8 key variables over the "
            "1096 days of the DE-Zrk daily series (2016-2018). `n` is the number of days "
            "with data; mean and median are in each variable's unit (NEE in gC m⁻² d⁻¹, "
            "FCH₄ in nmol m⁻² s⁻¹, TS in °C); skewness and excess kurtosis equal 0 for a "
            "normal distribution.",
        ),
        interpretacion=tr(
            "El NEE tiene media casi nula (0.05 gC m⁻² d⁻¹) pero mediana positiva "
            "(0.28): el sitio es fuente la mayor parte del ano y sumidero en pulsos "
            "intensos de verano. El FCH₄ es fuertemente asimetrico (media 80 frente "
            "a mediana 23 nmol m⁻² s⁻¹), tipico de emisiones de metano en turbera.",
            "NEE has a near-zero mean (0.05 gC m⁻² d⁻¹) but a positive median (0.28): the "
            "site is a source most of the year and a sink in intense summer pulses. FCH₄ "
            "is strongly skewed (mean 80 versus median 23 nmol m⁻² s⁻¹), typical of "
            "peatland methane emissions.",
        ),
        formato={c: "{:.3f}" for c in desc.columns if c not in ("variable", "n")},
    )

    I.tabla(
        A.outliers(),
        explicabilidad=tr(
            "Recuento de valores extremos por el criterio de rango intercuartilico: se "
            "marcan los que caen fuera de Q1 − k·IQR y Q3 + k·IQR, con k = 1.5 (criterio "
            "habitual) y k = 3.0 (solo extremos severos). `pct_1.5` y `pct_3.0` son el "
            "porcentaje de dias marcados. La ultima columna cuenta cuantos valores "
            "recortaria una winsorizacion a k = 3.",
            "Count of extreme values by the interquartile range criterion: values outside "
            "Q1 − k·IQR and Q3 + k·IQR are flagged, with k = 1.5 (usual criterion) and "
            "k = 3.0 (severe extremes only). `pct_1.5` and `pct_3.0` are the percentage "
            "of flagged days. The last column counts how many values a winsorization at "
            "k = 3 would clip.",
        ),
        interpretacion=tr(
            "Con el criterio IQR 1.5 el NEE tiene un 8.85 % de extremos y el FCH₄ un "
            "1.37 %. No se eliminan: son senal geofisica real (pulsos de emision, olas "
            "de calor) que el gemelo digital debe reproducir, asi que el entrenamiento "
            "usa el parquet original y no el winsorizado.",
            "With the IQR 1.5 criterion NEE has 8.85 % extremes and FCH₄ 1.37 %. They are "
            "not removed: they are real geophysical signal (emission pulses, heat waves) "
            "that the digital twin must reproduce, so training uses the original parquet "
            "and not the winsorized one.",
        ),
        formato={"pct_1.5": "{:.2f}", "pct_3.0": "{:.2f}"},
    )

    st.markdown(tr("#### Matriz de correlacion", "#### Correlation matrix"))
    metodo = st.radio(
        tr("Metodo", "Method"), ["pearson", "spearman"], horizontal=True,
        key="mia_corr_metodo",
        help=tr("Pearson mide relacion lineal; Spearman, monotona (robusta a los extremos).",
                "Pearson measures linear relationship; Spearman, monotonic (robust to extremes)."),
    )
    corr = A.correlacion(metodo)
    if corr.empty:
        I.falta_artefacto(tr(f"la matriz de correlacion ({metodo})",
                             f"the correlation matrix ({metodo})"), _CMD_EDA)
    else:
        I.grafico(
            charts.heatmap(corr, dominio=(-1, 1), titulo_valor="r"),
            explicabilidad=tr(
                f"Matriz de correlacion de {metodo.capitalize()} entre las variables "
                "clave. Cada celda es el coeficiente r del par fila-columna, de −1 "
                "(relacion inversa perfecta) a +1 (directa perfecta), con 0 en el "
                "centro de la escala de color. Pearson mide relacion LINEAL; Spearman "
                "opera sobre los rangos y capta cualquier relacion monotona, lo que la "
                "hace robusta a los extremos del FCH₄. La diagonal vale 1 por "
                "definicion y la matriz es simetrica.",
                f"{metodo.capitalize()} correlation matrix between the key variables. "
                "Each cell is the coefficient r of the row-column pair, from −1 (perfect "
                "inverse relationship) to +1 (perfect direct one), with 0 at the center "
                "of the color scale. Pearson measures LINEAR relationship; Spearman works "
                "on ranks and captures any monotonic relationship, which makes it robust "
                "to FCH₄ extremes. The diagonal equals 1 by definition and the matrix is "
                "symmetric.",
            ),
            interpretacion=tr(
                f"Correlacion de {metodo.capitalize()} entre las 15 variables clave. "
                "El FCH₄ va de la mano de la temperatura del suelo, y el NEE se explica "
                "sobre todo por la radiacion: es la senal que despues recoge la "
                "importancia de variables del modelo ganador. Ninguna pareja de drivers "
                "llega a colinealidad extrema, asi que se conservan las 32 features.",
                f"{metodo.capitalize()} correlation between the 15 key variables. FCH₄ "
                "goes hand in hand with soil temperature, and NEE is explained mostly by "
                "radiation: it is the signal later picked up by the variable importance "
                "of the winning model. No pair of drivers reaches extreme collinearity, "
                "so all 32 features are kept.",
            ),
        )

    I.tabla(
        A.normalidad(),
        explicabilidad=tr(
            "Dos contrastes de normalidad sobre cada serie: Shapiro-Wilk (estadistico W, "
            "tanto mas cercano a 1 cuanto mas normal) y D'Agostino-Pearson (K², que "
            "combina asimetria y curtosis). La hipotesis nula de ambos es que los datos "
            "proceden de una normal, asi que un p pequeno la RECHAZA. Se incluyen "
            "tambien los residuos de la descomposicion STL, para separar la no "
            "normalidad propia de la variable de la que induce la estacionalidad.",
            "Two normality tests on each series: Shapiro-Wilk (W statistic, closer to 1 "
            "the more normal) and D'Agostino-Pearson (K², combining skewness and "
            "kurtosis). The null hypothesis of both is that the data come from a normal "
            "distribution, so a small p REJECTS it. The residuals of the STL "
            "decomposition are also included, to separate the variable's own "
            "non-normality from the one induced by seasonality.",
        ),
        interpretacion=tr(
            "Shapiro-Wilk y D'Agostino rechazan la normalidad en las 10 series "
            "(p < 1e-25 en los objetivos). De ahi que la comparacion entre modelos use "
            "tests NO parametricos —Wilcoxon, Friedman, Nemenyi— y no una t de Student.",
            "Shapiro-Wilk and D'Agostino reject normality in all 10 series (p < 1e-25 "
            "for the targets). Hence the model comparison uses NON-parametric tests "
            "—Wilcoxon, Friedman, Nemenyi— and not a Student t.",
        ),
        formato={"shapiro_W": "{:.4f}", "shapiro_p": "{:.2e}",
                 "dagostino_K2": "{:.2f}", "dagostino_p": "{:.2e}",
                 "asimetria": "{:.3f}", "curtosis_exceso": "{:.3f}"},
    )

    st.markdown(tr("#### Figuras del EDA", "#### EDA figures"))
    izq, der = st.columns(2)
    with izq:
        I.figura(
            A.EDA_FIG / "07_descomposicion_STL_NEE.png",
            titulo_fig=tr("Descomposicion STL del NEE", "STL decomposition of NEE"),
            explicabilidad=tr(
                "Descomposicion STL (Seasonal-Trend decomposition using LOESS) de la "
                "serie de NEE en tres paneles apilados: tendencia (movimiento lento "
                "plurianual), componente estacional (el ciclo que se repite cada ano) y "
                "residuo (lo que no explica ninguna de las dos). Las tres se suman para "
                "reconstruir la serie original, y comparar sus amplitudes dice cual "
                "manda.",
                "STL decomposition (Seasonal-Trend decomposition using LOESS) of the NEE "
                "series in three stacked panels: trend (slow multi-year movement), "
                "seasonal component (the cycle repeated every year) and residual (what "
                "neither explains). The three add up to the original series, and "
                "comparing their amplitudes tells which one dominates.",
            ),
            interpretacion=tr(
                "La componente estacional domina sobre la tendencia: el balance de "
                "carbono lo manda el ciclo anual, no una deriva plurianual.",
                "The seasonal component dominates over the trend: the carbon balance is "
                "driven by the annual cycle, not by a multi-year drift.",
            ),
            comando=_CMD_EDA,
        )
    with der:
        I.figura(
            A.EDA_FIG / "09_climatologia_dia_del_anio.png",
            titulo_fig=tr("Climatologia del dia del ano", "Day-of-year climatology"),
            explicabilidad=tr(
                "Valor medio de cada driver para cada uno de los 366 dias del ano, "
                "promediando los tres anos de registro con una ventana circular de 15 "
                "dias (circular: el 31 de diciembre es vecino del 1 de enero). El eje X "
                "es el dia del ano, no una fecha concreta.",
                "Mean value of each driver for each of the 366 days of the year, "
                "averaging the three years of record with a circular 15-day window "
                "(circular: 31 December neighbors 1 January). The X axis is the day of "
                "the year, not a specific date.",
            ),
            interpretacion=tr(
                "Es la curva que usa el simulador para los drivers que el usuario no "
                "fija: media 2016-2018 con ventana circular de 15 dias.",
                "It is the curve the simulator uses for the drivers the user does not "
                "set: 2016-2018 mean with a circular 15-day window.",
            ),
            comando=_CMD_EDA,
        )

    I.texto_md(A.resumen_md("eda"))


# ================================================= 2 · Entrenamiento (historico)
def _entrenamiento() -> None:
    I.titulo(tr("Entrenamiento de los 5 modelos", "Training of the 5 models"),
             ayuda=tr("Ficha informativa de como se entrenaron. Para reentrenar de verdad, "
                      "ve a la subseccion «Reentrenar».",
                      "Information sheet on how they were trained. To really retrain, go "
                      "to the «Retrain» subsection."))

    comp = A.comparativa()
    if comp.empty:
        I.falta_artefacto("resultados_comparativa.csv", _CMD_TRAIN)
        return

    I.metricas(
        [
            (tr("Modelos", "Models"), len(comp)),
            ("Features", "32"),
            ("Train (2016-17)", tr("702 dias", "702 days")),
            ("Test (2018)", tr("365 dias", "365 days")),
        ],
        explicabilidad=tr(
            "Dimensiones del experimento de entrenamiento: cuantos modelos se comparan, "
            "cuantas variables de entrada recibe cada uno tras la ingenieria de features "
            "(retardos, medias moviles y terminos estacionales incluidos) y como se "
            "reparten los 1067 dias utiles entre el tramo de entrenamiento y el de test.",
            "Dimensions of the training experiment: how many models are compared, how "
            "many input variables each receives after feature engineering (lags, moving "
            "averages and seasonal terms included) and how the 1067 usable days are "
            "split between the training and the test segments.",
        ),
        interpretacion=tr(
            "El holdout es TEMPORAL, no aleatorio: se entrena con 2016-2017 y se valida "
            "con 2018 completo, sin barajar. Un split aleatorio filtraria el futuro "
            "hacia el pasado a traves de los lags y las medias moviles, e inflaria "
            "artificialmente el R².",
            "The holdout is TEMPORAL, not random: training uses 2016-2017 and validation "
            "the full 2018, unshuffled. A random split would leak the future into the "
            "past through the lags and moving averages, and would artificially inflate R².",
        ),
    )

    c_modelo, c_fam = tr("Modelo", "Model"), tr("Familia", "Family")
    c_tiempo = tr("Tiempo (s)", "Time (s)")
    ficha = pd.DataFrame.from_records(
        [
            {
                c_modelo: r["model"],
                c_fam: (tr("hibrido", "hybrid") if r["algorithm_type"] == "HYBRID"
                        else tr("clasico", "classical")),
                "Framework": r["framework"],
                c_tiempo: r["train_time_s"],
            }
            for _, r in comp.iterrows()
        ]
    )
    I.tabla(
        ficha,
        explicabilidad=tr(
            "Ficha de identidad de los cinco modelos: familia (clasico o hibrido), "
            "libreria con la que esta implementado y segundos de reloj que costo "
            "entrenarlo sobre el tramo 2016-2017. El tiempo es de una sola corrida en "
            "esta maquina, asi que vale para comparar ordenes de magnitud entre modelos, "
            "no como medida de rendimiento absoluto.",
            "Identity sheet of the five models: family (classical or hybrid), library "
            "it is implemented with and wall-clock seconds it took to train on the "
            "2016-2017 segment. The time comes from a single run on this machine, so it "
            "is good for comparing orders of magnitude between models, not as an "
            "absolute performance measure.",
        ),
        interpretacion=tr(
            "Tres clasicos (Random Forest, XGBoost, SVR/SVM) y dos hibridos (CNN-LSTM "
            "en Keras/TensorFlow y Stacking Ensemble). El coste de entrenamiento varia "
            "dos ordenes de magnitud, de 0.14 s del SVR a los ~18 s del Stacking, que "
            "entrena sus tres bases mas el meta-modelo.",
            "Three classical (Random Forest, XGBoost, SVR/SVM) and two hybrid (CNN-LSTM "
            "in Keras/TensorFlow and Stacking Ensemble). Training cost varies by two "
            "orders of magnitude, from 0.14 s for the SVR to ~18 s for the Stacking, "
            "which trains its three bases plus the meta-model.",
        ),
        formato={c_tiempo: "{:.2f}"},
    )

    st.markdown(tr(
        "**Por modelo se ajustan tres estimadores:** un regresor para NEE, otro para "
        "FCH₄ y un clasificador binario del balance (sumidero si NEE < 0). El "
        "desbalance de clases (~32 % sumidero) se trata con `class_weight=\"balanced\"` "
        "en RF y SVC, y con `scale_pos_weight` en XGBoost.",
        "**Three estimators are fitted per model:** a regressor for NEE, another for "
        "FCH₄ and a binary balance classifier (sink if NEE < 0). The class imbalance "
        "(~32 % sink) is handled with `class_weight=\"balanced\"` in RF and SVC, and "
        "with `scale_pos_weight` in XGBoost.",
    ))
    I.texto_md(A.resumen_md("entrenamiento"))


# ============================================================ 3 · Comparativa
def _comparativa() -> None:
    I.titulo(tr("Comparativa de algoritmos", "Algorithm comparison"),
             ayuda=tr("Metricas sobre el holdout de 2018, separadas por objetivo.",
                      "Metrics on the 2018 holdout, split by target."))

    comp = A.comparativa()
    if comp.empty:
        I.falta_artefacto("resultados_comparativa.csv", _CMD_TRAIN)
        return

    objetivo = st.radio(
        tr("Objetivo", "Target"), ["NEE (CO₂)", "FCH4 (CH₄)"], horizontal=True,
        key="mia_target",
    )
    pref = "nee" if objetivo.startswith("NEE") else "fch4"
    unidad = "gC m⁻² d⁻¹" if pref == "nee" else "nmol m⁻² s⁻¹"

    c_modelo, c_comb = tr("Modelo", "Model"), tr("Combinado", "Combined")
    tabla = pd.DataFrame.from_records(
        [
            {
                c_modelo: _NOMBRE_CORTO.get(r["model"], r["model"]),
                "RMSE": r[f"{pref}_rmse"],
                "MAE": r[f"{pref}_mae"],
                "R²": r[f"{pref}_r2"],
                "NSE": r[f"{pref}_nse"],
                "F1 macro": r["f1_macro"],
                "AUC": r["auc"],
                c_comb: r["combined_score"],
            }
            for _, r in comp.iterrows()
        ]
    ).sort_values(c_comb, ascending=False)

    I.tabla(
        tabla,
        explicabilidad=tr(
            f"Metricas de los cinco modelos para UN solo objetivo ({objetivo}), medidas "
            f"sobre el holdout de 2018 y ordenadas por score combinado. RMSE y MAE van "
            f"en {unidad} y son errores: menor es mejor. R² y NSE son adimensionales y "
            "valen 1 en la prediccion perfecta, 0 si el modelo iguala a predecir la "
            "media y negativo si la empeora. F1 macro y AUC son de la tarea de "
            "clasificacion, comunes a ambos objetivos, y por eso se repiten al cambiar "
            "el selector.",
            f"Metrics of the five models for a SINGLE target ({objetivo}), measured on "
            f"the 2018 holdout and sorted by combined score. RMSE and MAE are in "
            f"{unidad} and are errors: lower is better. R² and NSE are dimensionless and "
            "equal 1 for a perfect prediction, 0 if the model matches predicting the mean "
            "and negative if it does worse. F1 macro and AUC belong to the classification "
            "task, common to both targets, which is why they repeat when the selector "
            "changes.",
        ),
        interpretacion=tr(
            f"Metricas de {objetivo} en 2018. El R² del NEE se mueve entre 0.16 y 0.24 "
            "mientras el del FCH₄ llega a 0.84: un orden de magnitud de diferencia. "
            "El metano se explica bien con temperatura del suelo y nivel freatico; el "
            "NEE necesitaria indices de vegetacion (NDVI/LAI) que este dataset no tiene.",
            f"{objetivo} metrics in 2018. The NEE R² ranges between 0.16 and 0.24 while "
            "that of FCH₄ reaches 0.84: an order of magnitude of difference. Methane is "
            "well explained by soil temperature and water table; NEE would need "
            "vegetation indices (NDVI/LAI) that this dataset does not have.",
        ),
        formato={"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.3f}", "NSE": "{:.3f}",
                 "F1 macro": "{:.3f}", "AUC": "{:.3f}", c_comb: "{:.4f}"},
    )

    c_obj = tr("Objetivo", "Target")
    largo = pd.DataFrame.from_records(
        [
            {c_modelo: _NOMBRE_CORTO.get(r["model"], r["model"]),
             c_obj: etiqueta, "R²": r[f"{clave}_r2"]}
            for _, r in comp.iterrows()
            for clave, etiqueta in (("nee", "NEE (CO₂)"), ("fch4", "FCH4 (CH₄)"))
        ]
    )
    I.grafico(
        charts.barras_agrupadas(largo, x=c_modelo, y="R²", color=c_obj,
                                titulo_y=tr("R² en el holdout 2018", "R² on the 2018 holdout")),
        explicabilidad=tr(
            "Barras agrupadas: para cada modelo del eje X, dos barras con el R² que "
            "alcanza en NEE (CO₂) y en FCH4 (CH₄) sobre el holdout de 2018. El eje Y es "
            "adimensional y mas alto es mejor. Al estar ambos objetivos en la misma "
            "escala, la altura relativa de las dos barras de un mismo modelo es lo que "
            "hay que mirar.",
            "Grouped bars: for each model on the X axis, two bars with the R² it reaches "
            "on NEE (CO₂) and on FCH4 (CH₄) on the 2018 holdout. The Y axis is "
            "dimensionless and higher is better. Since both targets share the same scale, "
            "the relative height of the two bars of one model is what to look at.",
        ),
        interpretacion=tr(
            "La brecha entre objetivos se mantiene en los cinco algoritmos: no es un "
            "problema de modelo sino de informacion disponible. La excepcion es el "
            "CNN-LSTM, que ademas hunde el R² del FCH₄ a 0.46 porque la ventana de 30 "
            "dias suaviza los picos de emision.",
            "The gap between targets holds across the five algorithms: it is not a model "
            "problem but one of available information. The exception is the CNN-LSTM, "
            "which also drops the FCH₄ R² to 0.46 because the 30-day window smooths the "
            "emission peaks.",
        ),
    )

    c_metrica, c_valor = tr("Metrica", "Metric"), tr("Valor", "Value")
    I.grafico(
        charts.barras_agrupadas(
            pd.DataFrame.from_records(
                [{c_modelo: _NOMBRE_CORTO.get(r["model"], r["model"]),
                  c_metrica: m, c_valor: r[k]}
                 for _, r in comp.iterrows()
                 for m, k in (("F1 macro", "f1_macro"), ("AUC", "auc"))]
            ),
            x=c_modelo, y=c_valor, color=c_metrica,
            titulo_y=tr("Clasificacion sumidero/fuente", "Sink/source classification"),
        ),
        explicabilidad=tr(
            "Las dos metricas de clasificacion binaria sumidero/fuente para cada modelo. "
            "F1 macro promedia el F1 de las dos clases dandoles el mismo peso, de modo "
            "que la clase minoritaria no se diluye; AUC mide la capacidad de ordenar los "
            "dias por probabilidad de sumidero y vale 0.5 en el azar. Ambas van de 0 a 1 "
            "y mas alto es mejor.",
            "The two binary sink/source classification metrics for each model. F1 macro "
            "averages the F1 of the two classes giving them the same weight, so the "
            "minority class is not diluted; AUC measures the ability to rank days by sink "
            "probability and equals 0.5 at chance. Both range from 0 to 1 and higher is "
            "better.",
        ),
        interpretacion=tr(
            "En clasificacion los cinco quedan muy juntos (AUC 0.86-0.92). Por eso el "
            "criterio de seleccion pondera regresion y clasificacion al 50 % y usa "
            "F1 macro y AUC en vez de accuracy, que premiaria al clasificador trivial "
            "«siempre fuente» por el desbalance 32/68.",
            "In classification the five end up very close together (AUC 0.86-0.92). That "
            "is why the selection criterion weights regression and classification at 50 % "
            "and uses F1 macro and AUC instead of accuracy, which would reward the trivial "
            "«always source» classifier because of the 32/68 imbalance.",
        ),
    )

    I.figura(
        A.TRAIN_FIG / "heatmap_comparativo_metricas.png",
        titulo_fig=tr("Heatmap comparativo (todas las metricas normalizadas)",
                      "Comparative heatmap (all normalized metrics)"),
        explicabilidad=tr(
            "Una fila por modelo y una columna por metrica. Cada COLUMNA se normaliza "
            "min-max por separado, de forma que el mejor valor de esa metrica queda en "
            "verde y el peor en rojo con independencia de su magnitud real; en las "
            "metricas de error el sentido se invierte antes de colorear, para que verde "
            "signifique siempre «mejor». Los numeros impresos son los valores "
            "normalizados, no las metricas originales.",
            "One row per model and one column per metric. Each COLUMN is min-max "
            "normalized separately, so the best value of that metric is green and the "
            "worst red regardless of its real magnitude; for error metrics the direction "
            "is inverted before coloring, so green always means «better». The printed "
            "numbers are the normalized values, not the original metrics.",
        ),
        interpretacion=tr(
            "Muestra el ORDEN relativo, nunca la magnitud: un verde intenso puede "
            "corresponder a un R² mediocre si todos los modelos lo son en esa columna, "
            "que es justo lo que pasa con el NEE. Leida asi, la figura dice que el "
            "Stacking domina en CONJUNTO sin ser el mejor en casi ninguna metrica "
            "aislada, y esa es la razon de que haga falta un criterio de seleccion "
            "explicito en vez de elegir por la mejor columna.",
            "It shows the relative ORDER, never the magnitude: an intense green may "
            "correspond to a mediocre R² if all models are mediocre in that column, which "
            "is exactly what happens with NEE. Read that way, the figure says the Stacking "
            "dominates as a WHOLE without being the best on almost any single metric, and "
            "that is why an explicit selection criterion is needed instead of choosing by "
            "the best column.",
        ),
        comando=_CMD_TRAIN,
    )


# ============================================================= 4 · Prediccion
def _prediccion() -> None:
    I.titulo(tr("Prediccion interactiva", "Interactive prediction"),
             ayuda=tr("Fija las condiciones y predice NEE y FCH₄ con el modelo que elijas. "
                      "Los modelos se cargan de disco: no se reentrena nada.",
                      "Set the conditions and predict NEE and FCH₄ with the model you "
                      "choose. Models are loaded from disk: nothing is retrained."))

    modelos = A.modelos_en_disco()
    if not modelos:
        I.falta_artefacto(tr("los modelos entrenados", "the trained models"), _CMD_TRAIN)
        return

    if len(modelos) == 1:
        st.info(tr(
            "Solo hay un modelo en disco. Ejecuta "
            f"`{_CMD_TRAIN}` (o el boton de «Reentrenar») para guardar los cinco y "
            "poder compararlos aqui.",
            "There is only one model on disk. Run "
            f"`{_CMD_TRAIN}` (or the «Retrain» button) to save all five and be able to "
            "compare them here.",
        ))

    izq, der = st.columns([1, 2], gap="large")

    with izq:
        elegido = st.selectbox(
            tr("Modelo", "Model"), modelos, format_func=lambda m: f"{m.nombre} · {m.formato}",
            key="mia_modelo",
        )
        wtd = st.slider(tr("Nivel freatico objetivo (cm)", "Target water table (cm)"),
                        -20, 80, 20, 1, key="mia_wtd")
        fecha = st.date_input(tr("Fecha", "Date"), date(2019, 7, 15), format="YYYY-MM-DD",
                              key="mia_fecha")

        st.markdown(tr("**Drivers (opcional)**", "**Drivers (optional)**"))
        st.caption(tr("Lo que no marques se toma de la climatologia del dia del ano.",
                      "Whatever you do not tick is taken from the day-of-year climatology."))
        rangos = P.rangos_climatologia()
        overrides: dict[str, float] = {}
        for _, fila in rangos.iterrows():
            clave = fila["clave"]
            if st.checkbox(tr(f"Fijar {fila['etiqueta'].lower()}",
                              f"Set {fila['etiqueta'].lower()}"), key=f"mia_use_{clave}"):
                overrides[clave] = st.slider(
                    f"{fila['etiqueta']} ({fila['unidad']})",
                    float(fila["min"]), float(fila["max"]), float(fila["media"]),
                    key=f"mia_val_{clave}",
                )

    with der:
        try:
            entrada, supuestos = P.construir_fila(
                wtd_cm=float(wtd), on_date=fecha, overrides=overrides
            )
            res = P.predecir(elegido.ruta, entrada, supuestos)
        except FileNotFoundError as exc:
            st.error(tr(f"No se pudo cargar el modelo: {exc}", f"Could not load the model: {exc}"))
            return

        I.metricas(
            [
                ("CO₂ · NEE (gC m⁻² d⁻¹)", F.fmt_signed(res.nee, 2)),
                ("CH₄ · FCH4 (nmol m⁻² s⁻¹)", F.fmt(res.fch4, 1)),
                (tr("Balance", "Balance"), _clase(res.clase).capitalize()),
            ],
            explicabilidad=tr(
                "Salida del modelo elegido para las condiciones fijadas a la izquierda: "
                "flujo de CO₂ (NEE, en gC m⁻² d⁻¹, negativo = captura), flujo de CH₄ "
                "(FCH4, en nmol m⁻² s⁻¹, siempre positivo) y la clase de balance que "
                "predice el clasificador. Los drivers que no se hayan marcado proceden "
                "de la climatologia de ese dia del ano, no de una medida real.",
                "Output of the chosen model for the conditions set on the left: CO₂ flux "
                "(NEE, in gC m⁻² d⁻¹, negative = uptake), CH₄ flux (FCH4, in "
                "nmol m⁻² s⁻¹, always positive) and the balance class predicted by the "
                "classifier. Drivers that were not ticked come from the climatology of "
                "that day of the year, not from a real measurement.",
            ),
            interpretacion=tr(
                f"P(sumidero) = {F.fmt(res.proba_sumidero, 3)} segun el clasificador; "
                f"por el signo del NEE seria «{_clase(res.clase_por_signo)}». Cuando ambos "
                "coinciden la prediccion es solida; si discrepan, el dia esta cerca del "
                "punto de equilibrio y conviene tomarla con cautela.",
                f"P(sink) = {F.fmt(res.proba_sumidero, 3)} according to the classifier; "
                f"by the NEE sign it would be «{_clase(res.clase_por_signo)}». When both "
                "agree the prediction is solid; if they disagree, the day is close to the "
                "break-even point and should be taken with caution.",
            ),
        )
        with st.expander(tr("Supuestos de esta simulacion", "Assumptions of this simulation")):
            for s in res.supuestos:
                st.markdown(f"- {s}")


# =========================================================== 5 · Mejor modelo
def _mejor_modelo() -> None:
    I.titulo(tr("Mejor modelo", "Best model"),
             ayuda=tr("Detalle del modelo desplegado y por que gano.",
                      "Detail of the deployed model and why it won."))

    meta = A.metadata_modelo()
    if not meta:
        I.falta_artefacto("modelo_final_metadata.json", _CMD_TRAIN)
        return

    m = meta.get("metrics_holdout") or meta.get("metrics_winner") or {}

    # El nombre del ganador NO va en la tira de metricas. Una tira de metricas
    # es una fila de instrumentos: columnas iguales y valores en mono tabular,
    # que es lo correcto para cifras y lo peor posible para un nombre propio
    # largo. «Stacking Ensemble (tuned)» pedia 298px en una columna de 173px.
    # Va como encabezado, donde dispone del ancho completo.
    st.markdown(f"#### {meta.get('winner', '—')}")

    I.metricas(
        [
            (tr("Score combinado", "Combined score"), F.fmt(meta.get("combined_score"), 4)),
            ("R² NEE", F.fmt(m.get("nee_r2"), 3)),
            ("R² FCH₄", F.fmt(m.get("fch4_r2"), 3)),
            ("AUC", F.fmt(m.get("auc"), 3)),
        ],
        explicabilidad=tr(
            "Tarjeta del modelo desplegado, leida de `modelo_final_metadata.json`. El "
            "score combinado es el criterio unico con el que se eligio entre los cinco "
            "candidatos; los tres siguientes son sus metricas en el holdout de 2018, en "
            "bruto y sin normalizar: R² de cada objetivo y AUC de la clasificacion.",
            "Card of the deployed model, read from `modelo_final_metadata.json`. The "
            "combined score is the single criterion used to choose among the five "
            "candidates; the next three are its 2018 holdout metrics, raw and "
            "unnormalized: R² of each target and AUC of the classification.",
        ),
        interpretacion=tr(
            f"Criterio: `{meta.get('criterio', '—')}`. Mitad regresion (R² medio de los "
            "dos objetivos, normalizado min-max entre modelos) y mitad clasificacion "
            "(F1 macro y AUC). No se usa accuracy por el desbalance de clases.",
            f"Criterion: `{meta.get('criterio', '—')}`. Half regression (mean R² of the "
            "two targets, min-max normalized across models) and half classification "
            "(F1 macro and AUC). Accuracy is not used because of the class imbalance.",
        ),
    )

    pesos = A.pesos_meta()
    if not pesos.empty:
        I.grafico(
            charts.barras_horizontales(pesos, etiqueta="modelo_base", valor="peso",
                                       titulo_x=tr("Coeficiente del meta-modelo Ridge",
                                                   "Ridge meta-model coefficient"),
                                       altura=180),
            explicabilidad=tr(
                "Coeficientes que el meta-modelo Ridge asigna a cada uno de los tres "
                "modelos base del Stacking. El Ridge recibe como entrada las "
                "predicciones de las bases y aprende con que peso combinarlas; un "
                "coeficiente mayor significa que esa base influye mas en la prediccion "
                "final. Las barras van ordenadas de mayor a menor.",
                "Coefficients the Ridge meta-model assigns to each of the three Stacking "
                "base models. The Ridge takes the bases' predictions as input and learns "
                "what weight to combine them with; a larger coefficient means that base "
                "influences the final prediction more. Bars are sorted from largest to "
                "smallest.",
            ),
            interpretacion=tr(
                "Como combina el ensemble a sus tres bases. El Ridge reparte el peso "
                "casi por igual entre Random Forest y SVR y castiga a XGBoost: el "
                "Stacking gana por juntar errores poco correlacionados, no porque un "
                "modelo base domine.",
                "How the ensemble combines its three bases. The Ridge splits the weight "
                "almost equally between Random Forest and SVR and penalizes XGBoost: the "
                "Stacking wins by pooling weakly correlated errors, not because one base "
                "model dominates.",
            ),
        )

    imp = A.importancias_rf(top=15)
    if not imp.empty:
        I.grafico(
            charts.barras_horizontales(imp, etiqueta="variable", valor="importancia",
                                       titulo_x=tr("Importancia (Random Forest base)",
                                                   "Importance (base Random Forest)")),
            explicabilidad=tr(
                "Las 15 variables mas importantes segun el Random Forest base del "
                "ensemble. La importancia es la de scikit-learn por reduccion media de "
                "impureza: suma 1 entre todas las features y mide cuanto contribuye cada "
                "una a separar los nodos de los arboles. Es una medida RELATIVA y "
                "conocida por favorecer a las variables continuas con muchos valores "
                "distintos, asi que ordena bien pero no cuantifica un efecto causal.",
                "The 15 most important variables according to the ensemble's base Random "
                "Forest. The importance is scikit-learn's mean decrease in impurity: it "
                "sums to 1 across all features and measures how much each contributes to "
                "splitting the tree nodes. It is a RELATIVE measure and known to favor "
                "continuous variables with many distinct values, so it ranks well but does "
                "not quantify a causal effect.",
            ),
            interpretacion=tr(
                "Importancia del Random Forest base del ensemble —el Stacking como tal "
                "no expone importancias globales, asi que esto es una aproximacion. "
                "Manda la radiacion (NETRAD_F, SW_IN_F), seguida de la temperatura del "
                "suelo acumulada a 30 dias: memoria termica del suelo, coherente con la "
                "fisica de una turbera.",
                "Importance of the ensemble's base Random Forest —the Stacking itself "
                "does not expose global importances, so this is an approximation. "
                "Radiation dominates (NETRAD_F, SW_IN_F), followed by 30-day accumulated "
                "soil temperature: thermal memory of the soil, consistent with the "
                "physics of a peatland.",
            ),
        )

    tn, fp, fn, tp = (m.get(k) for k in ("tn", "fp", "fn", "tp"))
    if None not in (tn, fp, fn, tp):
        real_f, real_s = tr("Real fuente", "True source"), tr("Real sumidero", "True sink")
        pred_f, pred_s = tr("Pred. fuente", "Pred. source"), tr("Pred. sumidero", "Pred. sink")
        I.tabla(
            pd.DataFrame.from_records(
                [
                    {"": real_f, pred_f: tn, pred_s: fp},
                    {"": real_s, pred_f: fn, pred_s: tp},
                ]
            ),
            explicabilidad=tr(
                "Matriz de confusion del modelo desplegado sobre los 365 dias de 2018. "
                "Las filas son la clase real y las columnas la predicha, asi que la "
                "diagonal son los aciertos. La clase positiva es «sumidero» (NEE < 0), "
                "que es la minoritaria. Fuera de la diagonal: arriba a la derecha, dias "
                "fuente que el modelo llama sumidero (falsos positivos); abajo a la "
                "izquierda, dias sumidero que se le escapan (falsos negativos).",
                "Confusion matrix of the deployed model over the 365 days of 2018. Rows "
                "are the true class and columns the predicted one, so the diagonal holds "
                "the hits. The positive class is «sink» (NEE < 0), the minority one. Off "
                "the diagonal: top right, source days the model calls sink (false "
                "positives); bottom left, sink days that escape it (false negatives).",
            ),
            interpretacion=tr(
                f"De los {tp + fn} dias sumidero de 2018 detecta {tp} "
                f"(recall {tp / (tp + fn):.0%}), a cambio de {fp} falsas alarmas. Para "
                "un gemelo digital de restauracion interesa mas no perderse dias "
                "sumidero que evitar falsos positivos, asi que el balance es el correcto.",
                f"Of the {tp + fn} sink days of 2018 it detects {tp} "
                f"(recall {tp / (tp + fn):.0%}), at the price of {fp} false alarms. For a "
                "restoration digital twin, not missing sink days matters more than "
                "avoiding false positives, so the balance is the right one.",
            ),
        )

    izq, der = st.columns(2)
    with izq:
        I.figura(A.TRAIN_FIG / "matrices_confusion.png",
                 titulo_fig=tr("Matrices de confusion (holdout 2018)",
                               "Confusion matrices (2018 holdout)"),
                 explicabilidad=tr(
                     "Una matriz por modelo, todas sobre el mismo holdout de 2018 y con "
                     "la misma disposicion que la tabla de arriba: filas = clase real, "
                     "columnas = predicha, diagonal = aciertos. Debajo de cada matriz se "
                     "imprimen F1 y recall de la clase sumidero.",
                     "One matrix per model, all on the same 2018 holdout and with the "
                     "same layout as the table above: rows = true class, columns = "
                     "predicted, diagonal = hits. Below each matrix the F1 and recall of "
                     "the sink class are printed.",
                 ),
                 interpretacion=tr(
                     "Permite comparar de un vistazo COMO se equivoca cada modelo, no "
                     "solo cuanto. Con un 32 % de sumidero, un clasificador que "
                     "respondiera «fuente» siempre ya lograria ~68 % de accuracy sin "
                     "aprender nada, y por eso debajo van F1 y recall y no la accuracy. "
                     "Para restauracion el error caro son los falsos positivos: "
                     "sobreestiman el beneficio de una intervencion.",
                     "It allows comparing at a glance HOW each model errs, not only how "
                     "much. With 32 % sink, a classifier always answering «source» would "
                     "already reach ~68 % accuracy without learning anything, which is "
                     "why F1 and recall, not accuracy, are shown below. For restoration "
                     "the costly error is false positives: they overestimate the benefit "
                     "of an intervention.",
                 ),
                 comando=_CMD_TRAIN)
    with der:
        I.figura(A.TRAIN_FIG / "curvas_roc_comparacion.png",
                 titulo_fig=tr("Curvas ROC", "ROC curves"),
                 explicabilidad=tr(
                     "Las cinco curvas ROC sobre unos mismos ejes: tasa de falsos "
                     "positivos en X frente a tasa de verdaderos positivos (recall de "
                     "sumidero) en Y, recorriendo todos los umbrales de decision "
                     "posibles. La diagonal es el clasificador al azar y el AUC es el "
                     "area bajo cada curva.",
                     "The five ROC curves on the same axes: false positive rate on X "
                     "against true positive rate (sink recall) on Y, sweeping all possible "
                     "decision thresholds. The diagonal is the random classifier and the "
                     "AUC is the area under each curve.",
                 ),
                 interpretacion=tr(
                     "El AUC va de 0.85 (CNN-LSTM) a 0.92 (Stacking): las cinco curvas "
                     "quedan muy juntas y por encima de la diagonal. Al resumir TODOS "
                     "los umbrales, el AUC mide la capacidad de ordenar dias por "
                     "probabilidad de sumidero y no el acierto con el umbral 0.5 que usa "
                     "la matriz de confusion; un margen tan estrecho es lo que obliga a "
                     "comprobar en las pruebas estadisticas si la diferencia es real.",
                     "AUC ranges from 0.85 (CNN-LSTM) to 0.92 (Stacking): the five curves "
                     "stay very close and above the diagonal. Since it summarizes ALL "
                     "thresholds, AUC measures the ability to rank days by sink "
                     "probability and not the accuracy at the 0.5 threshold used by the "
                     "confusion matrix; such a narrow margin is what forces checking in "
                     "the statistical tests whether the difference is real.",
                 ),
                 comando=_CMD_TRAIN)


# ============================================================= 6 · Reentrenar
def _reentrenar() -> None:
    I.titulo(tr("Reentrenar el motor", "Retrain the engine"),
             ayuda=tr("Entrenamiento REAL y sincrono. Bloquea la pagina hasta terminar.",
                      "REAL, synchronous training. It blocks the page until done."))

    st.warning(tr(
        "**Antes de empezar, ten en cuenta que:**\n\n"
        "- Se sobrescriben `modelo_final.pkl`, la tabla comparativa y las figuras.\n"
        "- El modelo desplegado hoy es el **Stacking tuned (v1.1, R² NEE 0.236)**. Un "
        "reentrenamiento sin tuning deja el **Stacking base (R² NEE 0.216)**: es un "
        "paso atras hasta que vuelvas a ejecutar `python -m ml.training.tuning`.\n"
        "- No hay lock ni cola: es un entorno de un solo usuario. No lo lances si hay "
        "otra persona usando la app.",
        "**Before starting, keep in mind that:**\n\n"
        "- `modelo_final.pkl`, the comparison table and the figures are overwritten.\n"
        "- The model deployed today is the **tuned Stacking (v1.1, NEE R² 0.236)**. A "
        "retraining without tuning leaves the **base Stacking (NEE R² 0.216)**: a step "
        "back until you run `python -m ml.training.tuning` again.\n"
        "- There is no lock or queue: it is a single-user environment. Do not launch it "
        "if someone else is using the app.",
    ))

    etiquetas = list(R.ALGORITMOS)
    hay_tf = R.tensorflow_disponible()
    if not hay_tf:
        st.error(tr(
            "TensorFlow no esta instalado: el CNN-LSTM no se puede entrenar. "
            "`pip install tensorflow==2.21.0`",
            "TensorFlow is not installed: the CNN-LSTM cannot be trained. "
            "`pip install tensorflow==2.21.0`",
        ))
        etiquetas = [e for e in etiquetas if e not in R.REQUIERE_TENSORFLOW]

    seleccion = st.multiselect(tr("Algoritmos a entrenar", "Algorithms to train"), etiquetas,
                               default=etiquetas, key="mia_algos")
    izq, der = st.columns(2)
    guardar = izq.checkbox(tr("Guardar los modelos al terminar", "Save the models when finished"),
                           value=True, key="mia_guardar")
    registrar = der.checkbox(tr("Registrar en PostgreSQL", "Register in PostgreSQL"),
                             value=True, key="mia_bd")

    if not seleccion:
        st.info(tr("Selecciona al menos un algoritmo.", "Select at least one algorithm."))
        return

    if st.button(tr("🚀 Iniciar reentrenamiento", "🚀 Start retraining"), type="primary"):
        barra = st.progress(0)
        estado = st.empty()
        inicio = time.perf_counter()
        with st.spinner(tr("Reentrenando motor de IA...", "Retraining the AI engine...")):
            for paso in R.ejecutar(seleccion, guardar=guardar, registrar_bd=registrar):
                barra.progress(min(paso.porcentaje, 100))
                estado.write(paso.mensaje)
                if paso.resultado is not None:
                    st.session_state[_CLAVE_RESULTADO] = {
                        "tabla": paso.resultado,
                        "ganador": paso.ganador,
                        "log": paso.log,
                        "segundos": time.perf_counter() - inicio,
                        "guardado": guardar,
                    }
        st.cache_data.clear()   # la comparativa y las series leen ficheros nuevos
        seg = time.perf_counter() - inicio
        st.success(tr(f"Terminado en {seg:.0f} s.", f"Finished in {seg:.0f} s."))

    guardado = st.session_state.get(_CLAVE_RESULTADO)
    if guardado:
        res = guardado["tabla"]
        c_modelo, c_comb, c_gan = tr("Modelo", "Model"), tr("Combinado", "Combined"), tr("Ganador", "Winner")
        tabla = pd.DataFrame.from_records(
            [
                {
                    c_modelo: _NOMBRE_CORTO.get(r["model"], r["model"]),
                    "RMSE NEE": r["nee_rmse"], "MAE NEE": r["nee_mae"], "R² NEE": r["nee_r2"],
                    "RMSE FCH₄": r["fch4_rmse"], "MAE FCH₄": r["fch4_mae"],
                    "R² FCH₄": r["fch4_r2"],
                    c_comb: r["combined_score"],
                    c_gan: "★" if r["winner"] else "",
                }
                for _, r in res.iterrows()
            ]
        ).sort_values(c_comb, ascending=False)
        sin_guardar = tr(" No se guardo nada: los artefactos en disco siguen intactos.",
                         " Nothing was saved: the artifacts on disk remain intact.")
        I.tabla(
            tabla,
            explicabilidad=tr(
                "Resultado de la corrida que se acaba de lanzar desde este panel, no de "
                "los artefactos en disco. Una fila por modelo con los errores y el R² de "
                "cada objetivo por separado, el score combinado que decide y una estrella "
                "en el ganador. Ordenada por score combinado descendente.",
                "Result of the run just launched from this panel, not of the artifacts "
                "on disk. One row per model with the errors and R² of each target "
                "separately, the combined score that decides and a star on the winner. "
                "Sorted by combined score, descending.",
            ),
            interpretacion=tr(
                f"Resultado de la corrida ({guardado['segundos']:.0f} s). Ganador: "
                f"**{guardado['ganador']}**. Las metricas son de regresion por objetivo "
                "—RMSE, MAE y R² de NEE y FCH₄—; la clasificacion entra solo a traves "
                "del score combinado."
                + ("" if guardado["guardado"] else sin_guardar),
                f"Run result ({guardado['segundos']:.0f} s). Winner: "
                f"**{guardado['ganador']}**. The metrics are per-target regression "
                "—RMSE, MAE and R² of NEE and FCH₄—; classification enters only through "
                "the combined score."
                + ("" if guardado["guardado"] else sin_guardar),
            ),
            formato={c: "{:.3f}" for c in tabla.columns
                     if c not in (c_modelo, c_gan)},
        )
        if guardado.get("log"):
            st.caption(tr(f"Log de la corrida: `{guardado['log']}`",
                          f"Run log: `{guardado['log']}`"))
        if guardado["guardado"]:
            st.info(tr(
                "Para recuperar el ajuste fino de hiperparametros (v1.1) ejecuta ahora:\n\n"
                f"```bash\n{_CMD_TUNING}\n```",
                "To recover the hyperparameter fine-tuning (v1.1) run now:\n\n"
                f"```bash\n{_CMD_TUNING}\n```",
            ))


# ================================================================== 7 · Logs
def _logs() -> None:
    I.titulo(tr("Logs y trazabilidad del pipeline", "Pipeline logs and traceability"))

    proc = A.procedencia()
    desfasados = int((proc["Estado"] == "desfasado").sum())
    faltan = int((proc["Estado"] == "falta").sum())
    if desfasados:
        diag = tr(
            f"{desfasados} artefacto(s) son ANTERIORES al parquet del dataset: se "
            "calcularon con datos mas viejos que los actuales y conviene "
            "regenerarlos con el comando de su fila.",
            f"{desfasados} artifact(s) are OLDER than the dataset parquet: they were "
            "computed with older data than the current one and should be "
            "regenerated with the command in their row.",
        )
    elif faltan:
        diag = tr(
            f"Faltan {faltan} artefacto(s); las subsecciones que dependan de ellos "
            "mostraran el aviso correspondiente.",
            f"{faltan} artifact(s) are missing; the subsections that depend on them "
            "will show the corresponding notice.",
        )
    else:
        diag = tr(
            "Todas las etapas estan al dia: cada artefacto es posterior al parquet "
            "del dataset, asi que los resultados que muestra esta seccion "
            "corresponden a los datos actuales.",
            "All stages are up to date: every artifact is newer than the dataset "
            "parquet, so the results shown in this section correspond to the current "
            "data.",
        )
    I.tabla(
        proc,
        explicabilidad=tr(
            "Inventario de los artefactos que produce el pipeline: para cada uno, su "
            "ruta, la fecha del fichero y un estado calculado comparando esa fecha con "
            "la del parquet del dataset. «al dia» = posterior al parquet; «desfasado» = "
            "anterior, luego se calculo con datos mas viejos; «falta» = no esta en "
            "disco. La ultima columna da el comando que lo regenera.",
            "Inventory of the artifacts the pipeline produces: for each one, its path, "
            "the file date and a status computed by comparing that date with the dataset "
            "parquet's. «al dia» (up to date) = newer than the parquet; «desfasado» "
            "(stale) = older, so it was computed with older data; «falta» (missing) = "
            "not on disk. The last column gives the command that regenerates it.",
        ),
        interpretacion=diag,
    )

    st.markdown(tr("#### Logs de ejecucion", "#### Execution logs"))
    ficheros = A.logs_disponibles()
    if not ficheros:
        st.info(tr(
            "Todavia no hay logs. El pipeline empezo a escribirlos en `ml/logs/` a "
            "partir de este cambio: aparecen aqui en cuanto ejecutes cualquier etapa "
            "o pulses «Iniciar reentrenamiento».",
            "There are no logs yet. The pipeline started writing them to `ml/logs/` "
            "from this change on: they appear here as soon as you run any stage or "
            "press «Start retraining».",
        ))
        return

    elegido = st.selectbox(
        tr("Fichero", "File"), ficheros, key="mia_log",
        format_func=lambda p: f"{p.name} · {p.stat().st_size / 1024:,.0f} KB",
    )
    filtro = st.text_input(tr("Filtrar lineas que contengan", "Filter lines containing"),
                           key="mia_log_filtro")
    contenido = A.leer_log(elegido)
    if filtro:
        contenido = "\n".join(l for l in contenido.splitlines() if filtro.lower() in l.lower())
    st.code(contenido or tr("(sin lineas que coincidan)", "(no matching lines)"), language=None)
    I.nota(tr(
        "Salida completa de la etapa, tal como se escribio en consola. Es la traza que "
        "permite reconstruir con que datos y en que orden se genero cada artefacto.",
        "Full output of the stage, as written to the console. It is the trace that "
        "allows reconstructing with which data and in which order each artifact was "
        "generated.",
    ))


# ==================================================== 8 · Validacion cruzada
def _validacion_cruzada() -> None:
    I.titulo(tr("Validacion cruzada temporal", "Temporal cross-validation"),
             ayuda=tr("TimeSeriesSplit: cada fold entrena con el pasado y valida con el "
                      "futuro inmediato, nunca al reves.",
                      "TimeSeriesSplit: each fold trains on the past and validates on the "
                      "immediate future, never the other way round."))

    resumen = A.cv_resumen()
    por_fold = A.cv_por_fold()
    if resumen.empty or por_fold.empty:
        I.falta_artefacto(tr("los resultados de validacion cruzada",
                             "the cross-validation results"), _CMD_CV)
        return

    c_modelo = tr("Modelo", "Model")
    tabla = pd.DataFrame.from_records(
        [
            {
                c_modelo: _NOMBRE_CORTO.get(r["model"], r["model"]),
                "Folds": int(r["n_folds"]),
                "R² NEE": f"{r['r2_nee_mean']:.3f} ± {r['r2_nee_std']:.3f}",
                "R² FCH₄": f"{r['r2_fch4_mean']:.3f} ± {r['r2_fch4_std']:.3f}",
                "RMSE NEE": f"{r['rmse_nee_mean']:.3f} ± {r['rmse_nee_std']:.3f}",
                "F1 macro": f"{r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f}",
                "AUC": f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}",
            }
            for _, r in resumen.iterrows()
        ]
    )
    I.tabla(
        tabla,
        explicabilidad=tr(
            "Resumen de la validacion cruzada temporal: media ± desviacion tipica de "
            "cada metrica a lo largo de los folds de `TimeSeriesSplit`. Cada fold amplia "
            "la ventana de entrenamiento y desplaza la de test hacia adelante, sin "
            "barajar nunca. La media dice como de bien va el modelo; la desviacion, "
            "como de constante es de un periodo a otro.",
            "Summary of the temporal cross-validation: mean ± standard deviation of each "
            "metric across the `TimeSeriesSplit` folds. Each fold widens the training "
            "window and shifts the test window forward, never shuffling. The mean says "
            "how well the model does; the deviation, how consistent it is from one "
            "period to another.",
        ),
        interpretacion=tr(
            "Media ± desviacion tipica sobre los folds. La desviacion del R² de NEE "
            "(±0.15 a ±0.33) es MAYOR que la diferencia entre modelos: por eso la "
            "eleccion del ganador no puede basarse en comparar medias y exige los tests "
            "estadisticos de la ultima subseccion.",
            "Mean ± standard deviation across folds. The deviation of the NEE R² "
            "(±0.15 to ±0.33) is LARGER than the difference between models: that is why "
            "choosing the winner cannot rely on comparing means and requires the "
            "statistical tests of the last subsection.",
        ),
    )

    I.grafico(
        charts.boxplot_folds(
            pd.DataFrame.from_records(
                [{c_modelo: _NOMBRE_CORTO.get(r["model"], r["model"]),
                  "R² NEE": r["r2_nee"]} for _, r in por_fold.iterrows()]
            ),
            modelo=c_modelo, valor="R² NEE",
            titulo_y=tr("R² de NEE por fold", "NEE R² per fold"),
        ),
        explicabilidad=tr(
            "Un diagrama de caja por modelo con los R² de NEE obtenidos en cada fold. "
            "La caja abarca del primer al tercer cuartil, la linea interior es la "
            "mediana y los bigotes llegan al minimo y al maximo observados. Con pocos "
            "folds cada caja resume muy pocos puntos, asi que se lee como dispersion "
            "orientativa y no como una distribucion bien estimada.",
            "One box plot per model with the NEE R² obtained in each fold. The box spans "
            "the first to the third quartile, the inner line is the median and the "
            "whiskers reach the observed minimum and maximum. With few folds each box "
            "summarizes very few points, so it reads as indicative dispersion and not as "
            "a well-estimated distribution.",
        ),
        interpretacion=tr(
            "Dispersion real entre folds. Las cajas se solapan casi por completo: "
            "ningun algoritmo es consistentemente mejor en NEE a lo largo del tiempo.",
            "Real dispersion between folds. The boxes overlap almost completely: no "
            "algorithm is consistently better on NEE over time.",
        ),
    )

    folds = A.folds_disponibles()
    fold = st.selectbox(tr("Ver un fold concreto", "View a specific fold"), folds,
                        key="mia_fold", format_func=lambda f: f"Fold {f}")
    detalle = por_fold[por_fold["fold"] == fold]
    columnas = ["model", "rmse_nee", "r2_nee", "rmse_fch4", "r2_fch4", "f1_macro", "auc"]
    I.tabla(
        detalle[[c for c in columnas if c in detalle.columns]].rename(
            columns={"model": c_modelo, "rmse_nee": "RMSE NEE", "r2_nee": "R² NEE",
                     "rmse_fch4": "RMSE FCH₄", "r2_fch4": "R² FCH₄",
                     "f1_macro": "F1 macro", "auc": "AUC"}
        ),
        explicabilidad=tr(
            f"Metricas de los cinco modelos en el fold {fold} por separado, sin "
            "promediar. Permite ver de donde sale la media y la desviacion de la tabla "
            "anterior: cada fold corresponde a un tramo temporal distinto, con mas "
            "historico de entrenamiento cuanto mayor es su numero.",
            f"Metrics of the five models in fold {fold} on its own, unaveraged. It shows "
            "where the mean and deviation of the previous table come from: each fold is a "
            "different time span, with more training history the higher its number.",
        ),
        interpretacion=tr(
            f"Fold {fold} aislado. Los folds tempranos entrenan con menos historico, "
            "asi que sus metricas suelen ser peores: es el comportamiento esperado de "
            "un TimeSeriesSplit y no un fallo del modelo.",
            f"Fold {fold} on its own. Early folds train with less history, so their "
            "metrics tend to be worse: this is the expected behavior of a "
            "TimeSeriesSplit and not a model failure.",
        ),
        formato={c: "{:.3f}" for c in
                 ("RMSE NEE", "R² NEE", "RMSE FCH₄", "R² FCH₄", "F1 macro", "AUC")},
    )

    I.figura(A.TRAIN_FIG / "cv_boxplots_estabilidad.png",
             titulo_fig=tr("Estabilidad entre folds", "Stability across folds"),
             explicabilidad=tr(
                 "Version que genera el pipeline de la misma comparacion: un panel por "
                 "metrica y, dentro de cada uno, un diagrama de caja por modelo con sus "
                 "valores a lo largo de los folds. Frente al grafico interactivo de "
                 "arriba, que muestra solo el R² del NEE, esta trae todas las metricas a "
                 "la vez y entra tal cual en el informe.",
                 "Pipeline-generated version of the same comparison: one panel per metric "
                 "and, within each, a box plot per model with its values across the folds. "
                 "Unlike the interactive chart above, which shows only the NEE R², this one "
                 "brings all metrics at once and goes into the report as is.",
             ),
             interpretacion=tr(
                 "La altura de cada caja es la inestabilidad del modelo entre periodos. "
                 "Sirve para comprobar que el solapamiento no es cosa de una metrica "
                 "concreta: se repite en casi todas, que es el argumento de fondo para "
                 "no elegir ganador comparando medias.",
                 "The height of each box is the model's instability between periods. It "
                 "checks that the overlap is not specific to one metric: it repeats in "
                 "almost all of them, which is the underlying argument for not choosing a "
                 "winner by comparing means.",
             ),
             comando=_CMD_CV)


# ======================================================= 9 · Hiperparametros
def _hiperparametros() -> None:
    I.titulo(tr("Hiperparametros", "Hyperparameters"),
             ayuda=tr("Valores fijados en el ajuste fino (v1.1).",
                      "Values fixed in the fine-tuning (v1.1)."))

    meta = A.metadata_modelo() or {}
    params = meta.get("best_params") or {}
    if params:
        for bloque, titulo_bloque in (
            ("regresion", tr("Regresion (NEE y FCH₄)", "Regression (NEE and FCH₄)")),
            ("clasificacion", tr("Clasificacion sumidero/fuente", "Sink/source classification")),
        ):
            valores = params.get(bloque) or {}
            if not valores:
                continue
            st.markdown(f"**{titulo_bloque}**")
            I.tabla(
                pd.DataFrame.from_records(
                    [{tr("Hiperparametro", "Hyperparameter"): k,
                      tr("Valor", "Value"): str(v)} for k, v in valores.items()]
                ),
                explicabilidad=tr(
                    "Valores definitivos de los hiperparametros del bloque de "
                    f"{titulo_bloque.lower()}, tal como quedaron fijados tras el ajuste "
                    "fino y como se guardaron en `modelo_final_metadata.json`. No son "
                    "los valores probados durante la busqueda, sino los ganadores: el "
                    "espacio explorado esta en el resumen del tuning, al final de esta "
                    "subseccion.",
                    "Final hyperparameter values of the "
                    f"{titulo_bloque.lower()} block, as fixed after the fine-tuning and as "
                    "saved in `modelo_final_metadata.json`. They are not the values tried "
                    "during the search but the winners: the explored space is in the tuning "
                    "summary, at the end of this subsection.",
                ),
                interpretacion=(
                    tr(
                        "Arboles poco profundos y tasa de aprendizaje baja (0.03) en "
                        "XGBoost: con 702 dias de entrenamiento, limitar la capacidad evita "
                        "memorizar el ruido diario. El meta-modelo paso de `LinearRegression` "
                        "a `Ridge(alpha=10)` porque las tres bases estan correlacionadas y "
                        "sus coeficientes se disparaban sin regularizacion.",
                        "Shallow trees and a low learning rate (0.03) in XGBoost: with 702 "
                        "training days, limiting capacity avoids memorizing daily noise. The "
                        "meta-model went from `LinearRegression` to `Ridge(alpha=10)` because "
                        "the three bases are correlated and their coefficients blew up "
                        "without regularization.",
                    )
                    if bloque == "regresion" else
                    tr(
                        "El clasificador usa arboles aun mas cortos (max_depth 3) y una "
                        "regresion logistica como meta-modelo: la frontera sumidero/fuente "
                        "es esencialmente el signo del NEE, y no necesita mas capacidad.",
                        "The classifier uses even shorter trees (max_depth 3) and a logistic "
                        "regression as meta-model: the sink/source boundary is essentially "
                        "the sign of NEE, and needs no more capacity.",
                    )
                ),
            )
    else:
        I.falta_artefacto(tr("los hiperparametros del tuning", "the tuning hyperparameters"),
                          _CMD_TUNING)

    tun = A.tuning()
    if tun.empty:
        return

    c_modelo = tr("Modelo", "Model")
    c_base, c_tuned = tr("R² NEE base", "R² NEE base"), "R² NEE tuned"
    c_delta = "Δ R² NEE"
    c_cb, c_ct = tr("Combinado base", "Combined base"), tr("Combinado tuned", "Combined tuned")
    tun = tun.copy()
    tun["ajustado"] = tun["model"].str.contains(r"\(tuned\)")
    base = tun[~tun["ajustado"]].set_index("model")
    filas = []
    for _, r in tun[tun["ajustado"]].iterrows():
        original = r["model"].replace(" (tuned)", "")
        if original not in base.index:
            continue
        filas.append(
            {
                c_modelo: _NOMBRE_CORTO.get(original, original),
                c_base: base.loc[original, "nee_r2"],
                c_tuned: r["nee_r2"],
                c_delta: r["nee_r2"] - base.loc[original, "nee_r2"],
                c_cb: base.loc[original, "combined_score"],
                c_ct: r["combined_score"],
            }
        )
    if filas:
        I.tabla(
            pd.DataFrame.from_records(filas),
            explicabilidad=tr(
                "Comparacion directa de cada modelo antes y despues del ajuste de "
                "hiperparametros, emparejando cada version «(tuned)» con su original. "
                "La columna Δ es la diferencia tuned − base del R² de NEE, asi que un "
                "valor positivo indica mejora. Las dos ultimas columnas hacen lo mismo "
                "con el score combinado, que es el criterio que decide si el modelo "
                "ajustado sustituye al original.",
                "Direct comparison of each model before and after hyperparameter tuning, "
                "pairing each «(tuned)» version with its original. The Δ column is the "
                "tuned − base difference in NEE R², so a positive value indicates "
                "improvement. The last two columns do the same with the combined score, "
                "which is the criterion that decides whether the tuned model replaces the "
                "original.",
            ),
            interpretacion=tr(
                "Efecto real del tuning. En el Stacking el R² de NEE sube de 0.216 a "
                "0.236 (+0.020) y el score combinado de 0.907 a 0.935. En XGBoost el "
                "ajuste EMPEORA el NEE: no todo tuning mejora, y por eso el pipeline "
                "solo promueve el modelo ajustado si supera al original.",
                "Real effect of tuning. In the Stacking the NEE R² rises from 0.216 to "
                "0.236 (+0.020) and the combined score from 0.907 to 0.935. In XGBoost "
                "the tuning WORSENS NEE: not all tuning improves, which is why the "
                "pipeline only promotes the tuned model if it beats the original.",
            ),
            formato={c: "{:.4f}" for c in (c_base, c_tuned, c_delta, c_cb, c_ct)},
        )
    I.texto_md(A.resumen_md("tuning"))


# =================================================== 10 · Pruebas estadisticas
# Dos diccionarios en paralelo, con las mismas claves (ids de seccion del CSV): la
# explicabilidad dice que prueba es y que hipotesis contrasta; la interpretacion,
# que salio y que se concluye.
def _explicabilidad_seccion() -> dict[str, str]:
    return {
        "1-normalidad-RMSE": tr(
            "Shapiro-Wilk aplicado a dos cosas distintas: la distribucion de RMSE de cada "
            "modelo a lo largo de los 5 folds, y la de las diferencias pareadas frente al "
            "modelo de referencia. La hipotesis nula es que los datos son normales, asi que "
            "`normal_alpha05 = SI` significa que NO se rechaza. De este resultado depende la "
            "rama que toma el resto de la bateria: parametrica o no parametrica.",
            "Shapiro-Wilk applied to two different things: the RMSE distribution of each "
            "model across the 5 folds, and that of the paired differences against the "
            "reference model. The null hypothesis is that the data are normal, so "
            "`normal_alpha05 = SI` means it is NOT rejected. The branch taken by the rest of "
            "the battery depends on this result: parametric or non-parametric.",
        ),
        "1-normalidad-F1": tr(
            "El mismo contraste de Shapiro-Wilk sobre el F1 macro en vez del RMSE, con la "
            "misma lectura: W cerca de 1 y p alto indican compatibilidad con la normal.",
            "The same Shapiro-Wilk test on macro F1 instead of RMSE, with the same reading: "
            "W close to 1 and a high p indicate compatibility with normality.",
        ),
        "2-pareada-RMSE": tr(
            "Comparacion uno a uno del modelo de referencia (Stacking Ensemble) contra cada "
            "uno de los otros cuatro, emparejando fold a fold. Al haberse rechazado la "
            "normalidad se usa Wilcoxon signed-rank, que trabaja con los rangos de las "
            "diferencias y no con sus valores. `p_holm` corrige por las 4 comparaciones "
            "simultaneas mediante el metodo de Holm, y es la columna que hay que mirar.",
            "One-to-one comparison of the reference model (Stacking Ensemble) against each "
            "of the other four, pairing fold by fold. Since normality was rejected, the "
            "Wilcoxon signed-rank test is used, which works with the ranks of the "
            "differences and not their values. `p_holm` corrects for the 4 simultaneous "
            "comparisons using Holm's method, and is the column to look at.",
        ),
        "2-pareada-F1": tr(
            "La misma prueba de Wilcoxon con correccion de Holm, aplicada al F1 macro.",
            "The same Wilcoxon test with Holm correction, applied to macro F1.",
        ),
        "3-omnibus-RMSE": tr(
            "Dos contrastes globales que preguntan si ALGUN modelo difiere del resto, sin "
            "decir cual. Friedman es la version para medidas repetidas y respeta que los "
            "cinco modelos se evaluaron sobre los mismos folds; Kruskal-Wallis trata las "
            "muestras como independientes y por tanto ignora ese emparejamiento.",
            "Two global tests asking whether ANY model differs from the rest, without saying "
            "which. Friedman is the repeated-measures version and respects that the five "
            "models were evaluated on the same folds; Kruskal-Wallis treats the samples as "
            "independent and therefore ignores that pairing.",
        ),
        "3-omnibus-F1": tr(
            "Los mismos dos contrastes globales sobre el F1 macro.",
            "The same two global tests on macro F1.",
        ),
        "3b-nemenyi-posthoc-RMSE": tr(
            "Post-hoc que solo tiene sentido despues de un Friedman significativo: compara "
            "los 10 pares posibles usando la distribucion del rango studentizado y ajusta "
            "por multiplicidad. La ultima fila da la diferencia critica CD, la distancia "
            "minima entre rangos medios para declarar significativa una diferencia.",
            "Post-hoc that only makes sense after a significant Friedman: it compares the 10 "
            "possible pairs using the studentized range distribution and adjusts for "
            "multiplicity. The last row gives the critical difference CD, the minimum "
            "distance between mean ranks to declare a difference significant.",
        ),
        "4-diebold-mariano": tr(
            "Prueba pensada para comparar capacidad predictiva sobre una serie temporal. No "
            "usa los 5 folds sino los 365 errores DIARIOS del holdout de 2018, uno por dia. "
            "Contrasta si la perdida media de los dos modelos difiere, con varianza de largo "
            "plazo de Newey-West (que absorbe la autocorrelacion de los errores diarios) y "
            "correccion de Harvey-Leybourne-Newbold para muestra pequena.",
            "Test designed to compare predictive ability on a time series. It does not use "
            "the 5 folds but the 365 DAILY errors of the 2018 holdout, one per day. It tests "
            "whether the mean loss of the two models differs, with a Newey-West long-run "
            "variance (which absorbs the autocorrelation of daily errors) and the "
            "Harvey-Leybourne-Newbold small-sample correction.",
        ),
    }


def _interpretacion_seccion() -> dict[str, str]:
    return {
        "1-normalidad-RMSE": tr(
            "Shapiro-Wilk sobre los RMSE de los 5 folds y sobre sus diferencias. Con n = 5 "
            "la prueba tiene poca potencia, asi que no rechazar normalidad no la demuestra: "
            "es la razon de usar de todos modos tests no parametricos despues.",
            "Shapiro-Wilk on the RMSEs of the 5 folds and on their differences. With n = 5 "
            "the test has little power, so failing to reject normality does not prove it: "
            "that is why non-parametric tests are used afterwards anyway.",
        ),
        "1-normalidad-F1": tr(
            "Lo mismo para el F1 macro. La diferencia Stacking-Random Forest sale NO normal "
            "(p = 0.013), lo que descarta directamente una t pareada sobre esa comparacion.",
            "The same for macro F1. The Stacking-Random Forest difference comes out NOT "
            "normal (p = 0.013), which directly rules out a paired t-test for that "
            "comparison.",
        ),
        "2-pareada-RMSE": tr(
            "Wilcoxon pareado con correccion de Holm. Ningun p ajustado baja de 0.05: en "
            "RMSE el Stacking NO es significativamente mejor que ningun rival, aunque su "
            "media sea menor. Con 5 folds el p minimo alcanzable por Wilcoxon es 0.0625.",
            "Paired Wilcoxon with Holm correction. No adjusted p falls below 0.05: in RMSE "
            "the Stacking is NOT significantly better than any rival, even though its mean "
            "is lower. With 5 folds the minimum p attainable by Wilcoxon is 0.0625.",
        ),
        "2-pareada-F1": tr(
            "Misma prueba sobre F1 macro. Tampoco hay diferencias significativas: las "
            "ventajas del ensemble en clasificacion caben dentro del ruido entre folds.",
            "Same test on macro F1. There are no significant differences either: the "
            "ensemble's advantages in classification fit within the noise between folds.",
        ),
        "3-omnibus-RMSE": tr(
            "Friedman detecta diferencias globales en RMSE (p = 0.0037): al menos un "
            "algoritmo se comporta distinto. Kruskal-Wallis no las ve (p = 0.264) porque "
            "ignora que los folds estan emparejados y pierde potencia.",
            "Friedman detects global differences in RMSE (p = 0.0037): at least one "
            "algorithm behaves differently. Kruskal-Wallis does not see them (p = 0.264) "
            "because it ignores that the folds are paired and loses power.",
        ),
        "3-omnibus-F1": tr(
            "En F1 macro ni Friedman (p = 0.282) ni Kruskal-Wallis encuentran diferencias: "
            "en clasificacion los cinco modelos son estadisticamente equivalentes.",
            "On macro F1 neither Friedman (p = 0.282) nor Kruskal-Wallis finds differences: "
            "in classification the five models are statistically equivalent.",
        ),
        "3b-nemenyi-posthoc-RMSE": tr(
            "Post-hoc de Nemenyi tras el Friedman significativo. Solo separa a CNN-LSTM de "
            "Random Forest (p = 0.041) y de XGBoost (p = 0.023). El Stacking NO se distingue "
            "de RF ni de XGBoost: gana por su comportamiento CONJUNTO regresion + "
            "clasificacion, no por un RMSE mejor.",
            "Nemenyi post-hoc after the significant Friedman. It only separates CNN-LSTM "
            "from Random Forest (p = 0.041) and from XGBoost (p = 0.023). The Stacking is "
            "NOT distinguishable from RF or XGBoost: it wins through its JOINT regression + "
            "classification behavior, not through a better RMSE.",
        ),
        "4-diebold-mariano": tr(
            "Diebold-Mariano sobre los 365 errores diarios de 2018, con correccion de "
            "Newey-West a 7 dias por la autocorrelacion. Ni en NEE (p = 0.754) ni en FCH₄ "
            "(p = 0.137) la capacidad predictiva difiere de forma significativa: es la "
            "prueba mas exigente y confirma la conclusion anterior.",
            "Diebold-Mariano on the 365 daily errors of 2018, with a 7-day Newey-West "
            "correction for autocorrelation. Neither for NEE (p = 0.754) nor for FCH₄ "
            "(p = 0.137) does predictive ability differ significantly: it is the most "
            "demanding test and confirms the previous conclusion.",
        ),
    }


def _pruebas_estadisticas() -> None:
    I.titulo(tr("Pruebas estadisticas", "Statistical tests"),
             ayuda=tr("Contraste formal de si las diferencias entre modelos son reales.",
                      "Formal test of whether the differences between models are real."))

    df = A.pruebas_estadisticas()
    if df.empty:
        I.falta_artefacto("statistical_tests_resultados.csv", _CMD_STATS)
        return

    # Las secciones se leen del CSV, nunca se codifican a mano.
    secciones = A.secciones_estadisticas()
    elegida = st.selectbox(tr("Seccion", "Section"), secciones, key="mia_stat_seccion")
    bloque = df[df["seccion"] == elegida].dropna(axis=1, how="all")
    bloque = bloque.drop(columns=[c for c in ("seccion",) if c in bloque.columns])

    I.tabla(
        bloque,
        explicabilidad=_explicabilidad_seccion().get(
            elegida,
            tr(
                "Resultados del contraste tal como los escribio "
                "`ml/training/statistical_tests.py`. `p_value` es el p-valor crudo, `p_holm` "
                "el corregido por comparaciones multiples cuando aplica, y "
                "`significativo_alpha05` el veredicto al 5 %.",
                "Test results as written by `ml/training/statistical_tests.py`. `p_value` is "
                "the raw p-value, `p_holm` the one corrected for multiple comparisons when "
                "applicable, and `significativo_alpha05` the verdict at 5 %.",
            ),
        ),
        interpretacion=_interpretacion_seccion().get(
            elegida,
            tr(
                "Un p por debajo de 0.05 rechaza la hipotesis nula de que no hay "
                "diferencia. Con solo 5 folds la potencia es baja, asi que un resultado no "
                "significativo no demuestra equivalencia: demuestra que este diseno no "
                "puede distinguirlos.",
                "A p below 0.05 rejects the null hypothesis of no difference. With only 5 "
                "folds power is low, so a non-significant result does not prove "
                "equivalence: it shows that this design cannot tell them apart.",
            ),
        ),
        formato={c: "{:.4f}" for c in
                 ("statistic", "DM_stat", "shapiro_W", "p_value", "p_holm",
                  "mean_diff", "mean_loss_diff") if c in bloque.columns},
    )

    izq, der = st.columns(2)
    with izq:
        I.figura(A.TRAIN_FIG / "statistical_tests_nemenyi_cd.png",
                 titulo_fig=tr("Diagrama de diferencias criticas (Nemenyi)",
                               "Critical difference diagram (Nemenyi)"),
                 explicabilidad=tr(
                     "Los cinco modelos colocados sobre un eje de rango medio (1 = mejor "
                     "RMSE medio entre folds, 5 = peor). La diferencia critica CD es la "
                     "distancia minima entre dos rangos para que la diferencia sea "
                     "significativa al 5 %; las barras horizontales unen los grupos que "
                     "NO alcanzan esa distancia. Aqui CD = 2.728 con k = 5 modelos y "
                     "N = 5 folds.",
                     "The five models placed on a mean-rank axis (1 = best mean RMSE "
                     "across folds, 5 = worst). The critical difference CD is the minimum "
                     "distance between two ranks for the difference to be significant at "
                     "5 %; horizontal bars join the groups that do NOT reach that "
                     "distance. Here CD = 2.728 with k = 5 models and N = 5 folds.",
                 ),
                 interpretacion=tr(
                     "Los modelos unidos por una misma barra no son distinguibles entre "
                     "si. Con un CD de 2.728 sobre una escala que solo va de 1 a 5, casi "
                     "cualquier par queda unido: es la forma grafica de decir que cinco "
                     "folds no bastan para separar estos modelos. Solo CNN-LSTM se "
                     "despega de Random Forest y de XGBoost.",
                     "Models joined by the same bar are not distinguishable from each "
                     "other. With a CD of 2.728 on a scale that only spans 1 to 5, almost "
                     "any pair ends up joined: it is the graphical way of saying that five "
                     "folds are not enough to separate these models. Only CNN-LSTM "
                     "separates from Random Forest and XGBoost.",
                 ),
                 comando=_CMD_STATS)
    with der:
        I.figura(A.TRAIN_FIG / "statistical_tests_distribuciones.png",
                 titulo_fig=tr("Distribucion de las metricas por fold",
                               "Distribution of the metrics per fold"),
                 explicabilidad=tr(
                     "Las muestras sobre las que operan los contrastes: los valores de "
                     "RMSE y de F1 macro que cada modelo obtuvo en cada uno de los cinco "
                     "folds. Es el dato de partida, antes de cualquier prueba.",
                     "The samples the tests operate on: the RMSE and macro F1 values each "
                     "model obtained in each of the five folds. It is the starting data, "
                     "before any test.",
                 ),
                 interpretacion=tr(
                     "El solapamiento visual es la version grafica de por que "
                     "casi ningun contraste sale significativo.",
                     "The visual overlap is the graphical version of why almost no test "
                     "comes out significant.",
                 ),
                 comando=_CMD_STATS)

    I.texto_md(A.resumen_md("estadisticas"))


# ==================================================================== router
_VISTAS = {
    "eda": _eda,
    "entrenamiento": _entrenamiento,
    "comparativa": _comparativa,
    "prediccion": _prediccion,
    "mejor": _mejor_modelo,
    "reentrenar": _reentrenar,
    "logs": _logs,
    "cv": _validacion_cruzada,
    "hiperparametros": _hiperparametros,
    "estadisticas": _pruebas_estadisticas,
}


def render() -> None:
    st.title(tr("Motor IA", "AI Engine"))

    # Doble guarda: la pagina no se registra en la navegacion sin rol admin,
    # y ademas se comprueba aqui por si se llega por URL directa.
    if not auth.has_role("admin"):
        usuario = auth.current_user()
        roles = ", ".join(usuario.role_names) if usuario and usuario.role_names else tr("sin rol", "no role")
        st.warning(tr(
            f"**Acceso restringido.** Esta seccion es exclusiva del rol **admin**; tu "
            f"sesion actual es **{roles}**. Aqui se exponen la analitica del pipeline de "
            "ML y el reentrenamiento del motor.",
            f"**Restricted access.** This section is exclusive to the **admin** role; your "
            f"current session is **{roles}**. It exposes the ML pipeline analytics and the "
            "engine retraining.",
        ))
        return

    st.caption(tr(
        "Analitica del pipeline de ML sobre DE-Zrk 2016-2018. Salvo «Reentrenar», todas "
        "las subsecciones leen resultados ya calculados: no se entrena nada al abrirlas.",
        "ML pipeline analytics on DE-Zrk 2016-2018. Except «Retrain», all subsections "
        "read already-computed results: nothing is trained when opening them.",
    ))

    seccion = st.segmented_control(
        tr("Subseccion", "Subsection"), SECCIONES, default=SECCIONES[0], key=_CLAVE_SECCION,
        format_func=_etiqueta_seccion, label_visibility="collapsed",
    )
    st.divider()
    _VISTAS[seccion or SECCIONES[0]]()
