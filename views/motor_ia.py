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

# Prefijo `mia_` en las claves de sesion: identificables y barridas por el logout.
_CLAVE_SECCION = "mia_seccion"
_CLAVE_RESULTADO = "mia_resultado_reentrenamiento"

SECCIONES = [
    "🔍 EDA",
    "⚙️ Entrenamiento",
    "📊 Comparativa",
    "🔮 Prediccion",
    "🏆 Mejor modelo",
    "🔄 Reentrenar",
    "📋 Logs",
    "🔁 Validacion cruzada",
    "🎛️ Hiperparametros",
    "🧪 Pruebas estadisticas",
]

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
    I.titulo("Exploracion de datos (EDA)",
             ayuda="DE-Zrk (Zarnekow), serie diaria 2016-2018 · 1096 dias.")

    desc = A.descriptivos()
    if desc.empty:
        I.falta_artefacto("las tablas del EDA", _CMD_EDA)
        return

    I.tabla(
        desc,
        interpretacion=(
            "El NEE tiene media casi nula (0.05 gC m⁻² d⁻¹) pero mediana positiva "
            "(0.28): el sitio es fuente la mayor parte del ano y sumidero en pulsos "
            "intensos de verano. El FCH₄ es fuertemente asimetrico (media 80 frente "
            "a mediana 23 nmol m⁻² s⁻¹), tipico de emisiones de metano en turbera."
        ),
        formato={c: "{:.3f}" for c in desc.columns if c not in ("variable", "n")},
    )

    I.tabla(
        A.outliers(),
        interpretacion=(
            "Con el criterio IQR 1.5 el NEE tiene un 8.85 % de extremos y el FCH₄ un "
            "1.37 %. No se eliminan: son senal geofisica real (pulsos de emision, olas "
            "de calor) que el gemelo digital debe reproducir, asi que el entrenamiento "
            "usa el parquet original y no el winsorizado."
        ),
        formato={"pct_1.5": "{:.2f}", "pct_3.0": "{:.2f}"},
    )

    st.markdown("#### Matriz de correlacion")
    metodo = st.radio(
        "Metodo", ["pearson", "spearman"], horizontal=True, key="mia_corr_metodo",
        help="Pearson mide relacion lineal; Spearman, monotona (robusta a los extremos).",
    )
    corr = A.correlacion(metodo)
    if corr.empty:
        I.falta_artefacto(f"la matriz de correlacion ({metodo})", _CMD_EDA)
    else:
        I.grafico(
            charts.heatmap(corr, dominio=(-1, 1), titulo_valor="r"),
            interpretacion=(
                f"Correlacion de {metodo.capitalize()} entre las 15 variables clave. "
                "El FCH₄ va de la mano de la temperatura del suelo, y el NEE se explica "
                "sobre todo por la radiacion: es la senal que despues recoge la "
                "importancia de variables del modelo ganador. Ninguna pareja de drivers "
                "llega a colinealidad extrema, asi que se conservan las 32 features."
            ),
        )

    I.tabla(
        A.normalidad(),
        interpretacion=(
            "Shapiro-Wilk y D'Agostino rechazan la normalidad en las 10 series "
            "(p < 1e-25 en los objetivos). De ahi que la comparacion entre modelos use "
            "tests NO parametricos —Wilcoxon, Friedman, Nemenyi— y no una t de Student."
        ),
        formato={"shapiro_W": "{:.4f}", "shapiro_p": "{:.2e}",
                 "dagostino_K2": "{:.2f}", "dagostino_p": "{:.2e}",
                 "asimetria": "{:.3f}", "curtosis_exceso": "{:.3f}"},
    )

    st.markdown("#### Figuras del EDA")
    izq, der = st.columns(2)
    with izq:
        I.figura(
            A.EDA_FIG / "07_descomposicion_STL_NEE.png",
            titulo_fig="Descomposicion STL del NEE",
            interpretacion=(
                "La componente estacional domina sobre la tendencia: el balance de "
                "carbono lo manda el ciclo anual, no una deriva plurianual."
            ),
            comando=_CMD_EDA,
        )
    with der:
        I.figura(
            A.EDA_FIG / "09_climatologia_dia_del_anio.png",
            titulo_fig="Climatologia del dia del ano",
            interpretacion=(
                "Es la curva que usa el simulador para los drivers que el usuario no "
                "fija: media 2016-2018 con ventana circular de 15 dias."
            ),
            comando=_CMD_EDA,
        )

    I.texto_md(A.resumen_md("eda"))


# ================================================= 2 · Entrenamiento (historico)
def _entrenamiento() -> None:
    I.titulo("Entrenamiento de los 5 modelos",
             ayuda="Ficha informativa de como se entrenaron. Para reentrenar de verdad, "
                   "ve a la subseccion «Reentrenar».")

    comp = A.comparativa()
    if comp.empty:
        I.falta_artefacto("resultados_comparativa.csv", _CMD_TRAIN)
        return

    I.metricas(
        [
            ("Modelos", len(comp)),
            ("Features", "32"),
            ("Train (2016-17)", "702 dias"),
            ("Test (2018)", "365 dias"),
        ],
        interpretacion=(
            "El holdout es TEMPORAL, no aleatorio: se entrena con 2016-2017 y se valida "
            "con 2018 completo, sin barajar. Un split aleatorio filtraria el futuro "
            "hacia el pasado a traves de los lags y las medias moviles, e inflaria "
            "artificialmente el R²."
        ),
    )

    ficha = pd.DataFrame.from_records(
        [
            {
                "Modelo": r["model"],
                "Familia": "hibrido" if r["algorithm_type"] == "HYBRID" else "clasico",
                "Framework": r["framework"],
                "Tiempo (s)": r["train_time_s"],
            }
            for _, r in comp.iterrows()
        ]
    )
    I.tabla(
        ficha,
        interpretacion=(
            "Tres clasicos (Random Forest, XGBoost, SVR/SVM) y dos hibridos (CNN-LSTM "
            "en Keras/TensorFlow y Stacking Ensemble). El coste de entrenamiento varia "
            "dos ordenes de magnitud, de 0.14 s del SVR a los ~18 s del Stacking, que "
            "entrena sus tres bases mas el meta-modelo."
        ),
        formato={"Tiempo (s)": "{:.2f}"},
    )

    st.markdown(
        "**Por modelo se ajustan tres estimadores:** un regresor para NEE, otro para "
        "FCH₄ y un clasificador binario del balance (sumidero si NEE < 0). El "
        "desbalance de clases (~32 % sumidero) se trata con `class_weight=\"balanced\"` "
        "en RF y SVC, y con `scale_pos_weight` en XGBoost."
    )
    I.texto_md(A.resumen_md("entrenamiento"))


# ============================================================ 3 · Comparativa
def _comparativa() -> None:
    I.titulo("Comparativa de algoritmos",
             ayuda="Metricas sobre el holdout de 2018, separadas por objetivo.")

    comp = A.comparativa()
    if comp.empty:
        I.falta_artefacto("resultados_comparativa.csv", _CMD_TRAIN)
        return

    objetivo = st.radio(
        "Objetivo", ["NEE (CO₂)", "FCH4 (CH₄)"], horizontal=True, key="mia_target",
    )
    pref = "nee" if objetivo.startswith("NEE") else "fch4"
    unidad = "gC m⁻² d⁻¹" if pref == "nee" else "nmol m⁻² s⁻¹"

    tabla = pd.DataFrame.from_records(
        [
            {
                "Modelo": _NOMBRE_CORTO.get(r["model"], r["model"]),
                "RMSE": r[f"{pref}_rmse"],
                "MAE": r[f"{pref}_mae"],
                "R²": r[f"{pref}_r2"],
                "NSE": r[f"{pref}_nse"],
                "F1 macro": r["f1_macro"],
                "AUC": r["auc"],
                "Combinado": r["combined_score"],
            }
            for _, r in comp.iterrows()
        ]
    ).sort_values("Combinado", ascending=False)

    I.tabla(
        tabla,
        interpretacion=(
            f"Metricas de {objetivo} en 2018. El R² del NEE se mueve entre 0.16 y 0.24 "
            "mientras el del FCH₄ llega a 0.84: un orden de magnitud de diferencia. "
            "El metano se explica bien con temperatura del suelo y nivel freatico; el "
            "NEE necesitaria indices de vegetacion (NDVI/LAI) que este dataset no tiene."
        ),
        formato={"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.3f}", "NSE": "{:.3f}",
                 "F1 macro": "{:.3f}", "AUC": "{:.3f}", "Combinado": "{:.4f}"},
    )

    largo = pd.DataFrame.from_records(
        [
            {"Modelo": _NOMBRE_CORTO.get(r["model"], r["model"]),
             "Objetivo": etiqueta, "R²": r[f"{clave}_r2"]}
            for _, r in comp.iterrows()
            for clave, etiqueta in (("nee", "NEE (CO₂)"), ("fch4", "FCH4 (CH₄)"))
        ]
    )
    I.grafico(
        charts.barras_agrupadas(largo, x="Modelo", y="R²", color="Objetivo",
                                titulo_y="R² en el holdout 2018"),
        interpretacion=(
            "La brecha entre objetivos se mantiene en los cinco algoritmos: no es un "
            "problema de modelo sino de informacion disponible. La excepcion es el "
            "CNN-LSTM, que ademas hunde el R² del FCH₄ a 0.46 porque la ventana de 30 "
            "dias suaviza los picos de emision."
        ),
    )

    I.grafico(
        charts.barras_agrupadas(
            pd.DataFrame.from_records(
                [{"Modelo": _NOMBRE_CORTO.get(r["model"], r["model"]),
                  "Metrica": m, "Valor": r[k]}
                 for _, r in comp.iterrows()
                 for m, k in (("F1 macro", "f1_macro"), ("AUC", "auc"))]
            ),
            x="Modelo", y="Valor", color="Metrica", titulo_y="Clasificacion sumidero/fuente",
        ),
        interpretacion=(
            "En clasificacion los cinco quedan muy juntos (AUC 0.86-0.92). Por eso el "
            "criterio de seleccion pondera regresion y clasificacion al 50 % y usa "
            "F1 macro y AUC en vez de accuracy, que premiaria al clasificador trivial "
            "«siempre fuente» por el desbalance 32/68."
        ),
    )

    I.figura(
        A.TRAIN_FIG / "heatmap_comparativo_metricas.png",
        titulo_fig="Heatmap comparativo (todas las metricas normalizadas)",
        interpretacion=(
            "Verde = mejor, rojo = peor, normalizado min-max por columna. El Stacking "
            "domina en conjunto sin ser el mejor en ninguna metrica aislada."
        ),
        comando=_CMD_TRAIN,
    )


# ============================================================= 4 · Prediccion
def _prediccion() -> None:
    I.titulo("Prediccion interactiva",
             ayuda="Fija las condiciones y predice NEE y FCH₄ con el modelo que elijas. "
                   "Los modelos se cargan de disco: no se reentrena nada.")

    modelos = A.modelos_en_disco()
    if not modelos:
        I.falta_artefacto("los modelos entrenados", _CMD_TRAIN)
        return

    if len(modelos) == 1:
        st.info(
            "Solo hay un modelo en disco. Ejecuta "
            f"`{_CMD_TRAIN}` (o el boton de «Reentrenar») para guardar los cinco y "
            "poder compararlos aqui."
        )

    izq, der = st.columns([1, 2], gap="large")

    with izq:
        elegido = st.selectbox(
            "Modelo", modelos, format_func=lambda m: f"{m.nombre} · {m.formato}",
            key="mia_modelo",
        )
        wtd = st.slider("Nivel freatico objetivo (cm)", -20, 80, 20, 1, key="mia_wtd")
        fecha = st.date_input("Fecha", date(2019, 7, 15), format="YYYY-MM-DD",
                              key="mia_fecha")

        st.markdown("**Drivers (opcional)**")
        st.caption("Lo que no marques se toma de la climatologia del dia del ano.")
        rangos = P.rangos_climatologia()
        overrides: dict[str, float] = {}
        for _, fila in rangos.iterrows():
            clave = fila["clave"]
            if st.checkbox(f"Fijar {fila['etiqueta'].lower()}", key=f"mia_use_{clave}"):
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
            st.error(f"No se pudo cargar el modelo: {exc}")
            return

        I.metricas(
            [
                ("CO₂ · NEE (gC m⁻² d⁻¹)", F.fmt_signed(res.nee, 2)),
                ("CH₄ · FCH4 (nmol m⁻² s⁻¹)", F.fmt(res.fch4, 1)),
                ("Balance", res.clase.capitalize()),
            ],
            interpretacion=(
                f"P(sumidero) = {F.fmt(res.proba_sumidero, 3)} segun el clasificador; "
                f"por el signo del NEE seria «{res.clase_por_signo}». Cuando ambos "
                "coinciden la prediccion es solida; si discrepan, el dia esta cerca del "
                "punto de equilibrio y conviene tomarla con cautela."
            ),
        )
        with st.expander("Supuestos de esta simulacion"):
            for s in res.supuestos:
                st.markdown(f"- {s}")


# =========================================================== 5 · Mejor modelo
def _mejor_modelo() -> None:
    I.titulo("Mejor modelo", ayuda="Detalle del modelo desplegado y por que gano.")

    meta = A.metadata_modelo()
    if not meta:
        I.falta_artefacto("modelo_final_metadata.json", _CMD_TRAIN)
        return

    m = meta.get("metrics_holdout") or meta.get("metrics_winner") or {}
    I.metricas(
        [
            ("Ganador", meta.get("winner", "—")),
            ("Score combinado", F.fmt(meta.get("combined_score"), 4)),
            ("R² NEE", F.fmt(m.get("nee_r2"), 3)),
            ("R² FCH₄", F.fmt(m.get("fch4_r2"), 3)),
            ("AUC", F.fmt(m.get("auc"), 3)),
        ],
        interpretacion=(
            f"Criterio: `{meta.get('criterio', '—')}`. Mitad regresion (R² medio de los "
            "dos objetivos, normalizado min-max entre modelos) y mitad clasificacion "
            "(F1 macro y AUC). No se usa accuracy por el desbalance de clases."
        ),
    )

    pesos = A.pesos_meta()
    if not pesos.empty:
        I.grafico(
            charts.barras_horizontales(pesos, etiqueta="modelo_base", valor="peso",
                                       titulo_x="Coeficiente del meta-modelo Ridge",
                                       altura=180),
            interpretacion=(
                "Como combina el ensemble a sus tres bases. El Ridge reparte el peso "
                "casi por igual entre Random Forest y SVR y castiga a XGBoost: el "
                "Stacking gana por juntar errores poco correlacionados, no porque un "
                "modelo base domine."
            ),
        )

    imp = A.importancias_rf(top=15)
    if not imp.empty:
        I.grafico(
            charts.barras_horizontales(imp, etiqueta="variable", valor="importancia",
                                       titulo_x="Importancia (Random Forest base)"),
            interpretacion=(
                "Importancia del Random Forest base del ensemble —el Stacking como tal "
                "no expone importancias globales, asi que esto es una aproximacion. "
                "Manda la radiacion (NETRAD_F, SW_IN_F), seguida de la temperatura del "
                "suelo acumulada a 30 dias: memoria termica del suelo, coherente con la "
                "fisica de una turbera."
            ),
        )

    tn, fp, fn, tp = (m.get(k) for k in ("tn", "fp", "fn", "tp"))
    if None not in (tn, fp, fn, tp):
        I.tabla(
            pd.DataFrame.from_records(
                [
                    {"": "Real fuente", "Pred. fuente": tn, "Pred. sumidero": fp},
                    {"": "Real sumidero", "Pred. fuente": fn, "Pred. sumidero": tp},
                ]
            ),
            interpretacion=(
                f"De los {tp + fn} dias sumidero de 2018 detecta {tp} "
                f"(recall {tp / (tp + fn):.0%}), a cambio de {fp} falsas alarmas. Para "
                "un gemelo digital de restauracion interesa mas no perderse dias "
                "sumidero que evitar falsos positivos, asi que el balance es el correcto."
            ),
        )

    izq, der = st.columns(2)
    with izq:
        I.figura(A.TRAIN_FIG / "matrices_confusion.png",
                 titulo_fig="Matrices de confusion (holdout 2018)",
                 interpretacion="Clase positiva = sumidero, la minoritaria (~32 %).",
                 comando=_CMD_TRAIN)
    with der:
        I.figura(A.TRAIN_FIG / "curvas_roc_comparacion.png",
                 titulo_fig="Curvas ROC",
                 interpretacion="AUC entre 0.85 (CNN-LSTM) y 0.92 (Stacking).",
                 comando=_CMD_TRAIN)


# ============================================================= 6 · Reentrenar
def _reentrenar() -> None:
    I.titulo("Reentrenar el motor",
             ayuda="Entrenamiento REAL y sincrono. Bloquea la pagina hasta terminar.")

    st.warning(
        "**Antes de empezar, ten en cuenta que:**\n\n"
        "- Se sobrescriben `modelo_final.pkl`, la tabla comparativa y las figuras.\n"
        "- El modelo desplegado hoy es el **Stacking tuned (v1.1, R² NEE 0.236)**. Un "
        "reentrenamiento sin tuning deja el **Stacking base (R² NEE 0.216)**: es un "
        "paso atras hasta que vuelvas a ejecutar `python -m ml.training.tuning`.\n"
        "- No hay lock ni cola: es un entorno de un solo usuario. No lo lances si hay "
        "otra persona usando la app."
    )

    etiquetas = list(R.ALGORITMOS)
    hay_tf = R.tensorflow_disponible()
    if not hay_tf:
        st.error(
            "TensorFlow no esta instalado: el CNN-LSTM no se puede entrenar. "
            "`pip install tensorflow==2.21.0`"
        )
        etiquetas = [e for e in etiquetas if e not in R.REQUIERE_TENSORFLOW]

    seleccion = st.multiselect("Algoritmos a entrenar", etiquetas, default=etiquetas,
                               key="mia_algos")
    izq, der = st.columns(2)
    guardar = izq.checkbox("Guardar los modelos al terminar", value=True, key="mia_guardar")
    registrar = der.checkbox("Registrar en PostgreSQL", value=True, key="mia_bd")

    if not seleccion:
        st.info("Selecciona al menos un algoritmo.")
        return

    if st.button("🚀 Iniciar reentrenamiento", type="primary"):
        barra = st.progress(0)
        estado = st.empty()
        inicio = time.perf_counter()
        with st.spinner("Reentrenando motor de IA..."):
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
        st.success(f"Terminado en {time.perf_counter() - inicio:.0f} s.")

    guardado = st.session_state.get(_CLAVE_RESULTADO)
    if guardado:
        res = guardado["tabla"]
        tabla = pd.DataFrame.from_records(
            [
                {
                    "Modelo": _NOMBRE_CORTO.get(r["model"], r["model"]),
                    "RMSE NEE": r["nee_rmse"], "MAE NEE": r["nee_mae"], "R² NEE": r["nee_r2"],
                    "RMSE FCH₄": r["fch4_rmse"], "MAE FCH₄": r["fch4_mae"],
                    "R² FCH₄": r["fch4_r2"],
                    "Combinado": r["combined_score"],
                    "Ganador": "★" if r["winner"] else "",
                }
                for _, r in res.iterrows()
            ]
        ).sort_values("Combinado", ascending=False)
        I.tabla(
            tabla,
            interpretacion=(
                f"Resultado de la corrida ({guardado['segundos']:.0f} s). Ganador: "
                f"**{guardado['ganador']}**. Las metricas son de regresion por objetivo "
                "—RMSE, MAE y R² de NEE y FCH₄—; la clasificacion entra solo a traves "
                "del score combinado."
                + ("" if guardado["guardado"] else
                   " No se guardo nada: los artefactos en disco siguen intactos.")
            ),
            formato={c: "{:.3f}" for c in tabla.columns
                     if c not in ("Modelo", "Ganador")},
        )
        if guardado.get("log"):
            st.caption(f"Log de la corrida: `{guardado['log']}`")
        if guardado["guardado"]:
            st.info(
                "Para recuperar el ajuste fino de hiperparametros (v1.1) ejecuta ahora:\n\n"
                f"```bash\n{_CMD_TUNING}\n```"
            )


# ================================================================== 7 · Logs
def _logs() -> None:
    I.titulo("Logs y trazabilidad del pipeline")

    proc = A.procedencia()
    desfasados = int((proc["Estado"] == "desfasado").sum())
    faltan = int((proc["Estado"] == "falta").sum())
    if desfasados:
        diag = (f"{desfasados} artefacto(s) son ANTERIORES al parquet del dataset: se "
                "calcularon con datos mas viejos que los actuales y conviene "
                "regenerarlos con el comando de su fila.")
    elif faltan:
        diag = (f"Faltan {faltan} artefacto(s); las subsecciones que dependan de ellos "
                "mostraran el aviso correspondiente.")
    else:
        diag = ("Todas las etapas estan al dia: cada artefacto es posterior al parquet "
                "del dataset, asi que los resultados que muestra esta seccion "
                "corresponden a los datos actuales.")
    I.tabla(proc, interpretacion=diag)

    st.markdown("#### Logs de ejecucion")
    ficheros = A.logs_disponibles()
    if not ficheros:
        st.info(
            "Todavia no hay logs. El pipeline empezo a escribirlos en `ml/logs/` a "
            "partir de este cambio: aparecen aqui en cuanto ejecutes cualquier etapa "
            "o pulses «Iniciar reentrenamiento»."
        )
        return

    elegido = st.selectbox(
        "Fichero", ficheros, key="mia_log",
        format_func=lambda p: f"{p.name} · {p.stat().st_size / 1024:,.0f} KB",
    )
    filtro = st.text_input("Filtrar lineas que contengan", key="mia_log_filtro")
    contenido = A.leer_log(elegido)
    if filtro:
        contenido = "\n".join(l for l in contenido.splitlines() if filtro.lower() in l.lower())
    st.code(contenido or "(sin lineas que coincidan)", language=None)
    I.nota(
        "Salida completa de la etapa, tal como se escribio en consola. Es la traza que "
        "permite reconstruir con que datos y en que orden se genero cada artefacto."
    )


# ==================================================== 8 · Validacion cruzada
def _validacion_cruzada() -> None:
    I.titulo("Validacion cruzada temporal",
             ayuda="TimeSeriesSplit: cada fold entrena con el pasado y valida con el "
                   "futuro inmediato, nunca al reves.")

    resumen = A.cv_resumen()
    por_fold = A.cv_por_fold()
    if resumen.empty or por_fold.empty:
        I.falta_artefacto("los resultados de validacion cruzada", _CMD_CV)
        return

    tabla = pd.DataFrame.from_records(
        [
            {
                "Modelo": _NOMBRE_CORTO.get(r["model"], r["model"]),
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
        interpretacion=(
            "Media ± desviacion tipica sobre los folds. La desviacion del R² de NEE "
            "(±0.15 a ±0.33) es MAYOR que la diferencia entre modelos: por eso la "
            "eleccion del ganador no puede basarse en comparar medias y exige los tests "
            "estadisticos de la ultima subseccion."
        ),
    )

    I.grafico(
        charts.boxplot_folds(
            pd.DataFrame.from_records(
                [{"Modelo": _NOMBRE_CORTO.get(r["model"], r["model"]),
                  "R² NEE": r["r2_nee"]} for _, r in por_fold.iterrows()]
            ),
            modelo="Modelo", valor="R² NEE", titulo_y="R² de NEE por fold",
        ),
        interpretacion=(
            "Dispersion real entre folds. Las cajas se solapan casi por completo: "
            "ningun algoritmo es consistentemente mejor en NEE a lo largo del tiempo."
        ),
    )

    folds = A.folds_disponibles()
    fold = st.selectbox("Ver un fold concreto", folds, key="mia_fold",
                        format_func=lambda f: f"Fold {f}")
    detalle = por_fold[por_fold["fold"] == fold]
    columnas = ["model", "rmse_nee", "r2_nee", "rmse_fch4", "r2_fch4", "f1_macro", "auc"]
    I.tabla(
        detalle[[c for c in columnas if c in detalle.columns]].rename(
            columns={"model": "Modelo", "rmse_nee": "RMSE NEE", "r2_nee": "R² NEE",
                     "rmse_fch4": "RMSE FCH₄", "r2_fch4": "R² FCH₄",
                     "f1_macro": "F1 macro", "auc": "AUC"}
        ),
        interpretacion=(
            f"Fold {fold} aislado. Los folds tempranos entrenan con menos historico, "
            "asi que sus metricas suelen ser peores: es el comportamiento esperado de "
            "un TimeSeriesSplit y no un fallo del modelo."
        ),
        formato={c: "{:.3f}" for c in
                 ("RMSE NEE", "R² NEE", "RMSE FCH₄", "R² FCH₄", "F1 macro", "AUC")},
    )

    I.figura(A.TRAIN_FIG / "cv_boxplots_estabilidad.png",
             titulo_fig="Estabilidad entre folds",
             interpretacion="Version del pipeline de la misma comparacion, con todas "
                            "las metricas a la vez.",
             comando=_CMD_CV)


# ======================================================= 9 · Hiperparametros
def _hiperparametros() -> None:
    I.titulo("Hiperparametros", ayuda="Valores fijados en el ajuste fino (v1.1).")

    meta = A.metadata_modelo() or {}
    params = meta.get("best_params") or {}
    if params:
        for bloque, titulo_bloque in (("regresion", "Regresion (NEE y FCH₄)"),
                                      ("clasificacion", "Clasificacion sumidero/fuente")):
            valores = params.get(bloque) or {}
            if not valores:
                continue
            st.markdown(f"**{titulo_bloque}**")
            I.tabla(
                pd.DataFrame.from_records(
                    [{"Hiperparametro": k, "Valor": str(v)} for k, v in valores.items()]
                ),
                interpretacion=(
                    "Arboles poco profundos y tasa de aprendizaje baja (0.03) en "
                    "XGBoost: con 702 dias de entrenamiento, limitar la capacidad evita "
                    "memorizar el ruido diario. El meta-modelo paso de `LinearRegression` "
                    "a `Ridge(alpha=10)` porque las tres bases estan correlacionadas y "
                    "sus coeficientes se disparaban sin regularizacion."
                    if bloque == "regresion" else
                    "El clasificador usa arboles aun mas cortos (max_depth 3) y una "
                    "regresion logistica como meta-modelo: la frontera sumidero/fuente "
                    "es esencialmente el signo del NEE, y no necesita mas capacidad."
                ),
            )
    else:
        I.falta_artefacto("los hiperparametros del tuning", _CMD_TUNING)

    tun = A.tuning()
    if tun.empty:
        return

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
                "Modelo": _NOMBRE_CORTO.get(original, original),
                "R² NEE base": base.loc[original, "nee_r2"],
                "R² NEE tuned": r["nee_r2"],
                "Δ R² NEE": r["nee_r2"] - base.loc[original, "nee_r2"],
                "Combinado base": base.loc[original, "combined_score"],
                "Combinado tuned": r["combined_score"],
            }
        )
    if filas:
        I.tabla(
            pd.DataFrame.from_records(filas),
            interpretacion=(
                "Efecto real del tuning. En el Stacking el R² de NEE sube de 0.216 a "
                "0.236 (+0.020) y el score combinado de 0.907 a 0.935. En XGBoost el "
                "ajuste EMPEORA el NEE: no todo tuning mejora, y por eso el pipeline "
                "solo promueve el modelo ajustado si supera al original."
            ),
            formato={c: "{:.4f}" for c in
                     ("R² NEE base", "R² NEE tuned", "Δ R² NEE",
                      "Combinado base", "Combinado tuned")},
        )
    I.texto_md(A.resumen_md("tuning"))


# =================================================== 10 · Pruebas estadisticas
_INTERPRETACION_SECCION = {
    "1-normalidad-RMSE": (
        "Shapiro-Wilk sobre los RMSE de los 5 folds y sobre sus diferencias. Con n = 5 "
        "la prueba tiene poca potencia, asi que no rechazar normalidad no la demuestra: "
        "es la razon de usar de todos modos tests no parametricos despues."
    ),
    "1-normalidad-F1": (
        "Lo mismo para el F1 macro. La diferencia Stacking-Random Forest sale NO normal "
        "(p = 0.013), lo que descarta directamente una t pareada sobre esa comparacion."
    ),
    "2-pareada-RMSE": (
        "Wilcoxon pareado con correccion de Holm. Ningun p ajustado baja de 0.05: en "
        "RMSE el Stacking NO es significativamente mejor que ningun rival, aunque su "
        "media sea menor. Con 5 folds el p minimo alcanzable por Wilcoxon es 0.0625."
    ),
    "2-pareada-F1": (
        "Misma prueba sobre F1 macro. Tampoco hay diferencias significativas: las "
        "ventajas del ensemble en clasificacion caben dentro del ruido entre folds."
    ),
    "3-omnibus-RMSE": (
        "Friedman detecta diferencias globales en RMSE (p = 0.0037): al menos un "
        "algoritmo se comporta distinto. Kruskal-Wallis no las ve (p = 0.264) porque "
        "ignora que los folds estan emparejados y pierde potencia."
    ),
    "3-omnibus-F1": (
        "En F1 macro ni Friedman (p = 0.282) ni Kruskal-Wallis encuentran diferencias: "
        "en clasificacion los cinco modelos son estadisticamente equivalentes."
    ),
    "3b-nemenyi-posthoc-RMSE": (
        "Post-hoc de Nemenyi tras el Friedman significativo. Solo separa a CNN-LSTM de "
        "Random Forest (p = 0.041) y de XGBoost (p = 0.023). El Stacking NO se distingue "
        "de RF ni de XGBoost: gana por su comportamiento CONJUNTO regresion + "
        "clasificacion, no por un RMSE mejor."
    ),
    "4-diebold-mariano": (
        "Diebold-Mariano sobre los 365 errores diarios de 2018, con correccion de "
        "Newey-West a 7 dias por la autocorrelacion. Ni en NEE (p = 0.754) ni en FCH₄ "
        "(p = 0.137) la capacidad predictiva difiere de forma significativa: es la "
        "prueba mas exigente y confirma la conclusion anterior."
    ),
}


def _pruebas_estadisticas() -> None:
    I.titulo("Pruebas estadisticas",
             ayuda="Contraste formal de si las diferencias entre modelos son reales.")

    df = A.pruebas_estadisticas()
    if df.empty:
        I.falta_artefacto("statistical_tests_resultados.csv", _CMD_STATS)
        return

    # Las secciones se leen del CSV, nunca se codifican a mano.
    secciones = A.secciones_estadisticas()
    elegida = st.selectbox("Seccion", secciones, key="mia_stat_seccion")
    bloque = df[df["seccion"] == elegida].dropna(axis=1, how="all")
    bloque = bloque.drop(columns=[c for c in ("seccion",) if c in bloque.columns])

    I.tabla(
        bloque,
        interpretacion=_INTERPRETACION_SECCION.get(
            elegida,
            "Resultados del contraste. La columna `significativo_alpha05` indica si se "
            "rechaza la hipotesis nula al 5 %.",
        ),
        formato={c: "{:.4f}" for c in
                 ("statistic", "DM_stat", "shapiro_W", "p_value", "p_holm",
                  "mean_diff", "mean_loss_diff") if c in bloque.columns},
    )

    izq, der = st.columns(2)
    with izq:
        I.figura(A.TRAIN_FIG / "statistical_tests_nemenyi_cd.png",
                 titulo_fig="Diagrama de diferencias criticas (Nemenyi)",
                 interpretacion="Modelos unidos por la misma barra no son distinguibles "
                                "entre si. CD = 2.728 para k = 5 y N = 5.",
                 comando=_CMD_STATS)
    with der:
        I.figura(A.TRAIN_FIG / "statistical_tests_distribuciones.png",
                 titulo_fig="Distribucion de las metricas por fold",
                 interpretacion="El solapamiento visual es la version grafica de por que "
                                "casi ningun contraste sale significativo.",
                 comando=_CMD_STATS)

    I.texto_md(A.resumen_md("estadisticas"))


# ==================================================================== router
_VISTAS = {
    SECCIONES[0]: _eda,
    SECCIONES[1]: _entrenamiento,
    SECCIONES[2]: _comparativa,
    SECCIONES[3]: _prediccion,
    SECCIONES[4]: _mejor_modelo,
    SECCIONES[5]: _reentrenar,
    SECCIONES[6]: _logs,
    SECCIONES[7]: _validacion_cruzada,
    SECCIONES[8]: _hiperparametros,
    SECCIONES[9]: _pruebas_estadisticas,
}


def render() -> None:
    st.title("Motor IA")

    # Doble guarda: la pagina no se registra en la navegacion sin rol admin,
    # y ademas se comprueba aqui por si se llega por URL directa.
    if not auth.has_role("admin"):
        usuario = auth.current_user()
        roles = ", ".join(usuario.role_names) if usuario and usuario.role_names else "sin rol"
        st.warning(
            f"**Acceso restringido.** Esta seccion es exclusiva del rol **admin**; tu "
            f"sesion actual es **{roles}**. Aqui se exponen la analitica del pipeline de "
            "ML y el reentrenamiento del motor."
        )
        return

    st.caption(
        "Analitica del pipeline de ML sobre DE-Zrk 2016-2018. Salvo «Reentrenar», todas "
        "las subsecciones leen resultados ya calculados: no se entrena nada al abrirlas."
    )

    seccion = st.segmented_control(
        "Subseccion", SECCIONES, default=SECCIONES[0], key=_CLAVE_SECCION,
        label_visibility="collapsed",
    )
    st.divider()
    _VISTAS[seccion or SECCIONES[0]]()
