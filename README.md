# PeatTwin — Gemelo Digital de Turberas

Monorepo para un **Gemelo Digital** orientado a la **restauración hidrológica de
turberas** y la **optimización del balance de carbono CO₂–CH₄**, construido sobre
datos reales de flujo de carbono del sitio **DE-Zrk (Zarnekow)**.

Cadena completa, extremo a extremo:

1. **ETL** de flujos FLUXNET-CH4 → parquet procesado (`ml/etl/`).
2. **EDA** con descriptivos, outliers (IQR), correlación, descomposición STL y
   normalidad Shapiro-Wilk (`ml/eda/`).
3. **Entrenamiento y comparación de 5 modelos** (Random Forest, XGBoost, SVR/SVM,
   CNN-LSTM y Stacking Ensemble), holdout temporal 2018 (`ml/training/`).
4. **Validación cruzada** temporal (`TimeSeriesSplit`, nº de folds configurable),
   **tuning** de hiperparámetros y **pruebas estadísticas** (Shapiro, Wilcoxon +
   Holm, Friedman, post-hoc de Nemenyi, Diebold-Mariano).
5. **Esquema PostgreSQL** vía Alembic, con capa de dominio en Python (modelos
   SQLAlchemy, servicios, predictor que carga el modelo desplegado sin
   reentrenar) y **generación de informes** PDF / Word / Excel.
6. **Aplicación Streamlit** con login y RBAC, series observado vs predicho,
   comparación de modelos, simulador de escenarios y administración de
   usuarios/roles.

> **Una sola aplicación, sin Docker y sin API HTTP.** Streamlit habla
> directamente con PostgreSQL en el mismo proceso: no hay backend REST, ni
> CORS, ni tokens, ni Node. Todo corre nativo en la máquina local con un `venv`
> de Python y **PostgreSQL instalado como servicio del sistema** (no en
> contenedor). No hay `Dockerfile` ni `docker-compose` en el repositorio.

---

## Estructura del repositorio

```
peattwin/
├── streamlit_app.py            # entrypoint: login + st.navigation (4 páginas)
├── views/                      # una vista por página
│   ├── login.py                #   entrar · crear cuenta
│   ├── series.py               #   series observado vs predicho
│   ├── modelos.py              #   comparación de modelos + informes
│   ├── simulador.py            #   simulador de escenarios
│   └── admin.py                #   usuarios y roles (solo admin)
├── ui/                         # capa de presentación compartida
│   ├── db.py                   #   session_scope() sobre SQLAlchemy
│   ├── auth.py                 #   sesión en st.session_state + RBAC
│   ├── charts.py               #   gráficos Altair
│   └── format.py               #   formateo de números y fechas
│
├── app/                        # dominio (sin dependencias de framework web)
│   ├── core/                   #   config.py · database.py · security.py · errors.py
│   ├── models/ · schemas/      #   SQLAlchemy + Pydantic
│   ├── services/               #   auth · site · flux · scenario · model · prediction
│   │                           #     · rbac · access (comprobaciones de rol/permiso)
│   ├── ml/predictor.py         #   carga el modelo activo (.pkl/.h5) SIN reentrenar
│   └── reports/                #   PDF (reportlab) · Word (python-docx) · Excel (openpyxl)
│
├── alembic/ · alembic.ini      # migraciones (0001_initial, 0002_digital_twin_schema)
├── requirements.txt · .env.example
│
├── ml/                         # venv propio, separado del de la app
│   ├── etl/                    #   run_etl.py, load_to_db.py  (FLUXNET-CH4 -> parquet + BD)
│   ├── eda/                    #   run_eda.py  (descriptivos, outliers, STL, Shapiro)
│   ├── training/               #   run_training.py, cross_validation.py, tuning.py,
│   │                           #     statistical_tests.py, models_classical.py, model_cnn_lstm.py
│   └── models/                 #   modelo_final.pkl / .h5 + metadata (no versionado en git)
│
└── data/
    ├── raw/                    # CSV originales de FLUXNET-CH4 (DE-Zrk) — ver licencia abajo
    └── processed/              # parquet + JSON de metadatos generados por el ETL
```

---

## Fuentes de datos y licencia

Todos los datos de flujo proceden de la red **FLUXNET-CH4** (dataset comunitario
2020, producto `FLUXNET-CH4_DD` = agregación **diaria**), un esfuerzo coordinado
por AmeriFlux Management Project y la **Integrated Carbon Observation System
(ICOS)** a partir de redes regionales.

| Campo | Valor |
|---|---|
| Sitio | **DE-Zrk — Zarnekow** (NE de Alemania) |
| Coordenadas | 53.87594 N, 12.88901 E |
| Tipo | Fen (turbera minerotrófica) rehumedecida hidrológicamente en 2004–2005 · IGBP `WET` · Köppen `Dfb` |
| Responsable (PI) | Torsten Sachs (GFZ Potsdam) |
| Red de origen | **EuroFlux / ICOS** (`ORIGINAL_DATA_SOURCE = EuroFlux`) |
| Periodo del CSV | 2013–2018 (2191 días); el modelado usa 2016–2018 |
| Ficheros en `data/raw/` | `FLX_DE-Zrk_FLUXNET-CH4_DD_2013-2018_1-1.csv` (serie diaria) · `FLX_AA-Flx_CH4-META_*.csv` (metadatos de sitios) |
| **Licencia de datos** | **CC-BY-4.0** — Creative Commons Attribution 4.0 (campo `FLUXNET-CH4_DATA_POLICY = CCBY4.0` en el fichero de metadatos). Uso libre con atribución. |

**Atribución requerida.** Al reutilizar resultados basados en estos datos, citar
la colección FLUXNET-CH4 y al PI del sitio DE-Zrk. Referencias:

- Delwiche, K. B., et al. (2021). *FLUXNET-CH4: a global, multi-ecosystem dataset
  and analysis of methane seasonality from freshwater wetlands.* Earth Syst. Sci.
  Data 13, 3607–3689. <https://doi.org/10.5194/essd-13-3607-2021>
- Sitio DE-Zrk (Zarnekow), PI Torsten Sachs — red EuroFlux/ICOS.

El objetivo de regresión (`NEE_F_ANNOPTLM`, `FCH4_F_ANNOPTLM`) ya viene
gap-filled al 100 % por el equipo del sitio con una red neuronal (ANNOPTLM); el
pipeline **no implementa gap-filling propio**. El sitio **no midió humedad del
suelo**, por lo que no hay ninguna variable SWC en el dataset (limitación
documentada en el EDA y en el informe técnico).

---

## Requisitos previos (instalados de forma nativa)

| Herramienta   | Versión recomendada | Notas |
|---------------|---------------------|-------|
| Python        | 3.11 o 3.12         | 3.11–3.12 para la app; 3.12 para el módulo ML (TensorFlow no soporta 3.13) |
| PostgreSQL    | 15 o superior       | Servicio local en `localhost:5432`; `psql` y `createdb` en el `PATH` |

No hace falta Node.js: la interfaz ya no es una SPA, es Streamlit.

En Windows, si `psql` / `createdb` no están en el `PATH`, añade la carpeta `bin`
de PostgreSQL, por ejemplo:
`C:\Program Files\PostgreSQL\16\bin`

---

## 1. Base de datos PostgreSQL (local)

Con el servicio de PostgreSQL en ejecución, crea la base de datos:

```bash
createdb -U postgres peattwin
```

Alternativa con `psql`:

```bash
psql -U postgres -c "CREATE DATABASE peattwin;"
```

> Se usará el superusuario `postgres` de la instalación local. Si prefieres un
> rol dedicado:
> ```bash
> psql -U postgres -c "CREATE ROLE peattwin WITH LOGIN PASSWORD 'peattwin';"
> psql -U postgres -c "ALTER DATABASE peattwin OWNER TO peattwin;"
> ```
> y ajusta `POSTGRES_USER` / `POSTGRES_PASSWORD` en el `.env` de la raíz.

---

## 2. Aplicación — venv, dependencias, migraciones y arranque

Todos los comandos desde la **raíz del repositorio**. **Sin Docker**: es un
`venv` de Python y el servidor de desarrollo de Streamlit.

```bash
# 2.1 Crear el entorno virtual (Python 3.11 o 3.12)
python -m venv .venv

# 2.2 Activarlo
.\.venv\Scripts\Activate.ps1         # Windows PowerShell
#  .\.venv\Scriptsctivate.bat      # Windows CMD
#  source .venv/bin/activate          # Linux / macOS

# 2.3 Instalar dependencias  (Streamlit, Altair, SQLAlchemy, Alembic, bcrypt,
#     scikit-learn, xgboost, reportlab, python-docx, openpyxl, matplotlib, …)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2.4 Configuración: copiar la plantilla y ajustar la contraseña de PostgreSQL
Copy-Item .env.example .env           # Linux/macOS:  cp .env.example .env
#  Edita .env -> POSTGRES_PASSWORD=<tu clave local>

# 2.5 Crear el esquema: aplica las dos migraciones de Alembic
alembic upgrade head
#  -> crea sites, flux_observations, restoration_scenarios, ml_models,
#     ml_model_metrics, predictions, users, roles, permissions (+ tablas de unión)
#     y siembra los 3 roles (admin / investigador / visualizador) y 11 permisos.

# 2.6 Levantar la aplicación  (comando exacto)
streamlit run streamlit_app.py
```

- App:  http://localhost:8501

> **Cargar datos y modelos.** Las migraciones crean el esquema **vacío**. Para
> poblar el sitio DE-Zrk, sus 1096 observaciones de flujo y los 12 registros de
> `ml_models` / `ml_model_metrics` que consume el dashboard, ejecuta el módulo ML
> una vez (sección 4). El fichero `ml/models/modelo_final.pkl` también lo genera
> ese módulo; la app lo **carga** en la primera predicción (no lo entrena).

### Acceso y control por rol

El **primer** usuario que se registre (pestaña «Crear una cuenta») recibe el rol
`admin` y `is_superuser`; el resto entran como `visualizador` (solo lectura). Un
admin amplía los roles desde la vista «Usuarios y roles». Las contraseñas se
guardan con **bcrypt** en la tabla `users`.

> La sesión vive en `st.session_state`: **refrescar el navegador (F5) cierra la
> sesión** y hay que volver a entrar. Es el comportamiento estándar de Streamlit.

Toda la interfaz se adapta al rol (`admin` / `investigador` / `visualizador`):

1. **Series temporales** — CO₂/CH₄ observados vs predichos por el modelo activo
   (backtest sobre las variables reales de cada día).
2. **Comparación de modelos** — tabla RMSE/MAE/R²/NSE/KGE · accuracy/F1/AUC ·
   tiempo, las figuras de matriz de confusión, ROC y heatmap del Paso 5-6, y la
   descarga de los tres informes (PDF ejecutivo, Word técnico, Excel de datos).
3. **Simulador de escenarios** — mueve el nivel freático objetivo y ve la
   predicción de CO₂/CH₄ y la clase sumidero/fuente, con curva de respuesta.
   Requiere el permiso `predictions:run`; guardar escenarios, `scenarios:write`.
4. **Usuarios y roles** (solo `admin`) — asignación de roles y activación de
   cuentas. La página ni siquiera aparece en el menú para el resto de roles.

### Migraciones posteriores

Tras añadir o modificar modelos en `app/models/` (e importarlos en
`app/models/__init__.py`):

```bash
alembic revision --autogenerate -m "descripcion del cambio"
alembic upgrade head
```

---

## 4. Módulo ML — genera los datos, los modelos y las figuras

`venv` propio, **separado del de la app**, desde la **raíz del repositorio**. Es lo
que produce el parquet procesado, las filas de `flux_observations` / `ml_models`
en PostgreSQL y `ml/models/modelo_final.pkl`.

```bash
python -m venv ml/.venv
.\ml\.venv\Scripts\Activate.ps1          # Linux/macOS:  source ml/.venv/bin/activate
pip install -r ml/requirements.txt        # numpy, pandas, scikit-learn, xgboost,
                                          #   tensorflow, statsmodels, scikit-posthocs…

export PGPASSWORD=<clave>                 # o define POSTGRES_* como en el .env de la raíz

# 4.1 ETL: CSV FLUXNET-CH4 -> data/processed/*.parquet + JSON de metadatos
python -m ml.etl.run_etl
# 4.2 Cargar las observaciones en PostgreSQL (crea el sitio DE-Zrk + 1096 filas)
python -m ml.etl.load_to_db
# 4.3 EDA -> ml/eda/figuras/ + ml/eda/EDA_DE-Zrk_resumen.md
python -m ml.eda.run_eda
# 4.4 Entrena los 5 modelos, guarda ml/models/modelo_final.pkl y registra
#     ml_models / ml_model_metrics (fold = -1 = holdout 2018)
python -m ml.training.run_training
# 4.5 Validación cruzada temporal (nº de folds configurable)
python -m ml.training.cross_validation --folds 5
# 4.6 Tuning: si mejora, actualiza modelo_final.pkl y ml_models (v1.1-tuned)
python -m ml.training.tuning
# 4.7 Pruebas estadísticas -> PRUEBAS_ESTADISTICAS_resumen.md + CSV de p-values
python -m ml.training.statistical_tests
```

Detalle en [ml/etl/README.md](ml/etl/README.md), [ml/eda/README.md](ml/eda/README.md)
y [ml/training/README.md](ml/training/README.md).

---

## 5. Verificación: el modelo se **carga**, no se reentrena al iniciar la app

Comprobado en el código y en ejecución:

- **Arranque sin entrenamiento.** `streamlit run streamlit_app.py` levanta la app
  en segundos: el entrypoint solo registra las páginas y **no** importa ni carga
  ningún modelo al iniciar. Un `grep` de `.fit(` / `GridSearchCV` /
  `RandomizedSearchCV` / `Epoch` sobre `app/`, `views/` y `ui/` no devuelve
  **ninguna** llamada de entrenamiento (solo aparece en textos de documentación).
- **Carga perezosa y desde disco.** El modelo se carga la **primera** vez que se
  abre el simulador o la vista de series, no antes.
  `app/ml/predictor.py::_load()`:
  - `.pkl` → `joblib.load(path)` (deserializa el `StackingRegressor` +
    `StackingClassifier` de scikit-learn/XGBoost).
  - `.h5` → `tensorflow.keras.models.load_model(path)` + `joblib.load` de los
    escaladores (`modelo_final_scalers.pkl`). Import de TensorFlow perezoso: solo
    ocurre si el modelo activo es un CNN-LSTM.
  - Ninguna de las dos rutas llama a `.fit()`.
- **Ruta del fichero.** Se resuelve por orden: `MODEL_FILE` de `.env` → columna
  `file_path` de la fila `ml_models` con `is_active = true` → `ml/models/modelo_final.{pkl,h5}`.
- **Caché en memoria.** El objeto cargado se cachea por `(ruta, mtime)` con un
  `RLock` a nivel de módulo; Streamlit ejecuta los reruns como hilos de un único
  proceso, así que la caché se comparte entre sesiones. Medido en esta máquina:
  1ª predicción ≈ **3.3 s** (incluye leer el `.pkl` y los parquet);
  2ª predicción ≈ **0.15 s** (reutiliza el objeto en memoria, sin releer disco).
  Al reemplazar el fichero (cambia el `mtime`) o al activar otro modelo, la
  siguiente predicción lo recarga sola.

Reentrenar es un paso **manual y explícito** del módulo ML (sección 4), nunca de
la interfaz.

---

## 6. Arquitectura de los 5 modelos comparados

Objetivos: **regresión** de NEE (CO₂) y FCH4 (CH₄) —un modelo por objetivo— y
**clasificación** binaria de la clase de balance (sumidero si NEE < 0, fuente si
NEE ≥ 0). 32 features (meteo gap-filled, WTD con lags 1/3/7 d y medias móviles
7/30 d, `TS_promedio` del perfil de suelo y sus derivadas, y variables
estacionales `doy`/`month` + seno/coseno). Evaluación con **holdout temporal**:
entrenamiento 2016–2017 (702 días), test 2018 (365), sin barajar. Desbalance de
clases (~32 % sumidero) tratado con `class_weight="balanced"` (RF/SVC) y
`scale_pos_weight` (XGBoost).

| # | Modelo | Familia | Arquitectura |
|---|--------|---------|--------------|
| 1 | **Random Forest** | clásico | `RandomForestRegressor(n_estimators=400, max_depth=None)` para NEE y FCH4; `RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample")`. Sin escalado (invariante). |
| 2 | **Gradient Boosting (XGBoost)** | clásico | `XGBRegressor/Classifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist")`; el clasificador con `scale_pos_weight = n_fuente/n_sumidero`. Sin escalado. |
| 3 | **SVR / SVM** | clásico | `SVR(kernel="rbf", C=10, epsilon=0.1)` dentro de `Pipeline(StandardScaler → SVR)` y envuelto en `TransformedTargetRegressor` (escala también el objetivo). Clasificación: `Pipeline(StandardScaler → SVC(rbf, C=10, class_weight="balanced", probability=True))`. |
| 4 | **CNN-LSTM** (Keras/TensorFlow) | híbrido | Ventana de **30 días** × 32 features → `Conv1D(64, k=3, padding="causal")` ×2 → `MaxPooling1D(2)` → `LSTM(64, return_sequences)` → `LSTM(32)` → `Dropout(0.2)` → `Dense(32)` → **dos cabezas**: `Dense(2)` lineal (regresión de [NEE, FCH4] estandarizados) y `Dense(1)` sigmoide (clasificación). Pérdida `MSE + w·BCE` (w = 2), `Adam(1e-3)`, hasta 120 épocas con early-stopping y `sample_weight` por clase. |
| 5 | **Stacking Ensemble** | híbrido | `StackingRegressor` / `StackingClassifier` con **3 modelos base** (Random Forest, XGBoost y SVR/SVC) y un **meta-modelo lineal** (`LinearRegression` en regresión, `LogisticRegression(class_weight="balanced")` en clasificación) que aprende a ponderar sus predicciones. Meta-features out-of-fold con `KFold(5)` en bloques. **Es el modelo desplegado**; tras el tuning su meta-modelo de regresión pasa a `Ridge(α=10)` (versión `1.1-tuned`, `is_active = true`). |

**Criterio de selección** (paso de comparación): `combined = 0.5·reg_score_norm +
0.5·(0.5·F1_macro + 0.5·AUC)` — 50 % regresión (R² medio de NEE y FCH4,
normalizado min-max entre modelos) y 50 % clasificación (F1_macro y AUC, no
accuracy, por el desbalance). El Stacking Ensemble gana por su mejor
comportamiento **conjunto** y su menor variabilidad entre folds; en RMSE puro es
estadísticamente indistinguible de Random Forest y XGBoost (ver
`ml/training/PRUEBAS_ESTADISTICAS_resumen.md`).

---

## Resumen de comandos de arranque diario (sin Docker)

| Servicio   | Carpeta / terminal | Comando exacto | URL |
|------------|--------------------|----------------|-----|
| PostgreSQL | —                  | servicio del sistema (`pg_ctl` / servicios de Windows) | `localhost:5432` |
| PeatTwin   | raíz del repo (venv activo) | `streamlit run streamlit_app.py` | http://localhost:8501 |

Una sola terminal además de la base de datos. Primera vez, además:
`createdb -U postgres peattwin` → `alembic upgrade head` (en la raíz) → poblar
datos y modelos con el módulo ML (sección 4).

---

## Estado

Implementado y verificado: ETL + carga en BD · EDA · 5 modelos entrenados y
comparados · validación cruzada temporal · tuning · pruebas estadísticas ·
esquema PostgreSQL con Alembic · capa de dominio en Python (servicios,
predicción sin reentrenar, informes PDF/Word/Excel) · aplicación Streamlit con
las 4 vistas, login bcrypt y control de acceso por rol. Todo corre **nativo**,
en un único proceso y sin contenedores.

Pendiente / mejora futura: índices de vegetación (NDVI/LAI) para elevar el R² del
NEE, más sitios FLUXNET-CH4, tests automatizados y CI.

---

## Uso de inteligencia artificial generativa en el desarrollo

Este proyecto se desarrolló con asistencia de **Claude Code** (Anthropic), un
asistente de programación basado en modelos de lenguaje de gran tamaño, operado
de forma interactiva a lo largo de tres sesiones de desarrollo. Se declara aquí
por coherencia con la declaración equivalente del artículo científico asociado.

**Para qué se usó:**

- Implementación y refinamiento iterativo, bajo especificación e instrucción del
  autor, de los módulos `ml/etl/`, `ml/eda/` y `ml/training/`, del esquema de
  base de datos y sus migraciones, de la capa de dominio en `app/`, del módulo
  de informes y de la aplicación Streamlit (`streamlit_app.py`, `views/`, `ui/`).
- Redacción del borrador del artículo científico a partir de los artefactos
  numéricos que produce este repositorio.
- Extracción y síntesis de la bibliografía consultada para el marco teórico.

**Para qué NO se usó:**

- No se empleó para generar, simular, alterar ni seleccionar datos. Todos los
  valores de `ml/eda/tablas/`, `ml/training/*.csv` y las figuras de
  `ml/*/figuras/` proceden de ejecutar el código de este repositorio sobre las
  observaciones reales de FLUXNET-CH4 del sitio DE-Zrk descritas más arriba.

El diseño experimental, la elección de arquitecturas y del protocolo de
validación, los criterios de selección de modelos y la interpretación de los
resultados corresponden al autor, que ha revisado y verificado la totalidad del
contenido generado con asistencia y asume la responsabilidad plena sobre la
exactitud, la integridad y la originalidad del trabajo.

Los commits de este repositorio llevan la línea `Co-Authored-By` correspondiente
a esa asistencia.
