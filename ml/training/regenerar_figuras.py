"""Regenera las figuras de entrenamiento a C.FIG_DPI SIN tocar modelos ni tablas.

`run_training`, `cross_validation` y `statistical_tests` guardan sus figuras al
final de un entrenamiento completo, y ese entrenamiento sobrescribe los modelos
de `ml/models/`, los CSV y la base de datos. Para cambiar solo la resolucion no
hace falta nada de eso, asi que este script:

  * lee de los CSV ya generados todo lo que las figuras necesitan
    (comparativa, CV por fold, pruebas estadisticas), y
  * reentrena en memoria los cinco modelos UNICAMENTE para obtener las
    probabilidades de la curva ROC, que no se guardan en disco. Nada de lo que
    reentrena se persiste, y compara el AUC obtenido con el del CSV para avisar
    si la curva ya no corresponde a la tabla.

Uso:  python -m ml.training.regenerar_figuras
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.training import config as C
from ml.training import cross_validation as CV
from ml.training import data as D
from ml.training import run_training as RT
from ml.training import statistical_tests as ST

AUC_TOL = 0.005


def figuras_desde_csv() -> None:
    res = pd.read_csv(C.RESULTS_CSV)
    RT.figure_heatmap(res)
    RT.figure_bars(res)

    # la matriz de confusion solo necesita tn/fp/fn/tp y las dos metricas del titulo
    RT.figure_confusion(res.to_dict("records"), frames={}, cnn_true=np.array([]))

    per_fold = pd.read_csv(C.CV_PER_FOLD_CSV)
    np.random.seed(C.SEED)  # jitter de los puntos del boxplot: reproducible
    CV.boxplots(per_fold)

    reg, cls = ST.matrices(ST.load_from_csv())
    ST.figure(reg, cls)
    pmat, avg_ranks, cd, _lib = ST.nemenyi_posthoc(reg)
    ST.nemenyi_figures(pmat, avg_ranks, cd)


def curva_roc() -> None:
    """Reentrena en memoria solo para recuperar las probabilidades del holdout."""
    res = pd.read_csv(C.RESULTS_CSV).set_index("model")
    df, feats = D.build_dataset()
    frames = D.train_test_frames(df, feats)
    spw = float((frames["cls_train"] == 0).sum() / (frames["cls_train"] == 1).sum())

    rows = [RT.run_classical(n, frames, spw) for n in
            ["Random Forest", "Gradient Boosting (XGBoost)", "SVR / SVM", "Stacking Ensemble"]]
    rows.append(RT.run_cnn_lstm(df, feats))

    _X, _Yr, Yc, Sp, _Ts = D.make_sequences(df, feats, C.WINDOW)
    cnn_true = Yc[Sp == "test"].astype(int)

    print("\nComprobacion contra resultados_comparativa.csv:")
    desvios = []
    for r in rows:
        d = abs(float(r["auc"]) - float(res.loc[r["model"], "auc"]))
        desvios.append(d)
        print(f"  {r['model']:28s} AUC reentrenado={r['auc']:.4f}  "
              f"CSV={res.loc[r['model'], 'auc']:.4f}  |dif|={d:.4f}"
              f"{'  <-- NO COINCIDE' if d > AUC_TOL else ''}")
    if max(desvios) > AUC_TOL:
        print("AVISO: la curva ROC reentrenada no reproduce el CSV; "
              "no se sobrescribe curvas_roc_comparacion.png.")
        return
    RT.figure_roc(rows, frames, cnn_true)


def main() -> None:
    print(f"Regenerando figuras de entrenamiento a {C.FIG_DPI} dpi")
    figuras_desde_csv()
    curva_roc()


if __name__ == "__main__":
    main()
