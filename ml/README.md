# PeatTwin - Modulo ML

Modulo independiente de la aplicacion. Comparte los datos con el resto del
monorepo a traves de la carpeta `/data`.

```
ml/
  etl/             # ETL de datos reales FLUXNET-CH4 (sitio DE-Zrk / Zarnekow)
    config.py      #   parametros del pipeline
    run_etl.py     #   pasos 1-7: CSV crudo -> 2 parquet + JSON de metadatos
    load_to_db.py  #   carga el parquet de observaciones en flux_observations
    README.md      #   decisiones de limpieza y supuestos (evidencia para el informe)
  eda/             # analisis exploratorio del dataset procesado
    run_eda.py     #   descriptivos, outliers (IQR), correlacion, STL, normalidad
    figuras/       #   13 figuras .png
    tablas/        #   tablas .csv
    EDA_DE-Zrk_resumen.md  #  resumen interpretado para el informe
  training/        # entrenamiento, comparacion y ajuste de modelos
    run_training.py      # RF, XGBoost, SVR/SVM, CNN-LSTM, Stacking; holdout 2018
    cross_validation.py  # validacion cruzada temporal (TimeSeriesSplit, folds configurables)
    tuning.py            # ajuste de hiperparametros (GridSearchCV/RandomizedSearchCV + TimeSeriesSplit)
    statistical_tests.py # pruebas estadisticas: Shapiro, t/Wilcoxon, ANOVA/Kruskal/Friedman, Diebold-Mariano
    models_classical.py  # constructores RF/XGB/SVR + stacking
    model_cnn_lstm.py    # CNN-LSTM multi-cabeza (Keras/TensorFlow)
    figuras/             # matrices de confusion, ROC, heatmap, boxplots
    resultados_comparativa.csv  # tabla consolidada de metricas (holdout)
    cv_por_fold.csv / cv_resumen.csv  # metricas de validacion cruzada
    tuning_resultados.csv / TUNING_resumen.md  # resultados del ajuste
    statistical_tests_resultados.csv / PRUEBAS_ESTADISTICAS_resumen.md  # pruebas estadisticas
  src/
    config.py      # rutas (data/, ml/models/) y semilla aleatoria
    data.py        # carga de datasets + generador sintetico de ejemplo
    train.py       # entrenamiento baseline (LinearRegression) ejecutable
    evaluate.py    # evaluacion de un modelo guardado
  notebooks/
    01_exploracion_datos.ipynb
  models/          # modelos entrenados (.joblib / .keras); versionado ignorado
```

## ETL de datos reales (DE-Zrk)

```bash
python -m ml.etl.run_etl        # -> data/processed/DE-Zrk_*.parquet + _etl_metadata.json
python -m ml.etl.load_to_db     # -> tabla flux_observations (idempotente; esquema migrado)
```

Detalle completo en [ml/etl/README.md](etl/README.md).

## Análisis exploratorio (EDA)

```bash
python -m ml.eda.run_eda       # -> ml/eda/figuras/, ml/eda/tablas/, EDA_DE-Zrk_resumen.md
```

Detalle en [ml/eda/README.md](eda/README.md).

## Entrenamiento y comparación de modelos

```bash
python -m ml.training.run_training       # 5 modelos, holdout 2018, figuras, CSV, guarda ganador, registra en BD
python -m ml.training.cross_validation   # validacion cruzada temporal (TimeSeriesSplit, --folds N)
python -m ml.training.tuning             # ajuste de hiperparametros de RF / Stacking / XGBoost
python -m ml.training.statistical_tests  # pruebas estadisticas de comparacion de modelos
```

Detalle en [ml/training/README.md](training/README.md).

## Puesta en marcha (nativo, sin Docker)

Desde la **raiz del repositorio**:

```bash
python -m venv ml/.venv
# Windows PowerShell:
.\ml\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source ml/.venv/bin/activate

pip install -r ml/requirements.txt

# Entrenamiento de ejemplo (genera ml/models/baseline.joblib)
python -m ml.src.train

# Evaluacion
python -m ml.src.evaluate ml/models/baseline.joblib

# Notebooks
jupyter lab   # o: jupyter notebook
```

> El esqueleto usa un dataset sintetico. Sustituye `make_synthetic_dataset()`
> por `load_processed_dataset("data/processed/<archivo>")` cuando tengas datos
> reales. TensorFlow/Keras esta comentado en `requirements.txt`; actívalo
> cuando el modelo lo requiera.
