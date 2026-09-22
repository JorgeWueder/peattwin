"""Figuras generadas al vuelo para los informes (matplotlib, backend Agg)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from app.reports.common import HoldoutSeries  # noqa: E402

_EMERALD = "#059669"
_INK = "#18181b"
_ORANGE = "#ea580c"


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def flux_series_png(hs: HoldoutSeries) -> bytes:
    x = np.arange(len(hs.dates))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True)
    axes[0].plot(x, hs.nee_obs, color=_INK, lw=1.0, label="observado")
    axes[0].plot(x, hs.nee_pred, color=_EMERALD, lw=1.8, label="predicho")
    axes[0].axhline(0, color="#a1a1aa", lw=0.8, ls="--")
    axes[0].set_ylabel("NEE / CO₂\n(gC m⁻² d⁻¹)")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].plot(x, hs.fch4_obs, color=_INK, lw=1.0)
    axes[1].plot(x, hs.fch4_pred, color=_EMERALD, lw=1.8)
    axes[1].set_ylabel("FCH4 / CH₄\n(nmol m⁻² s⁻¹)")
    step = max(1, len(hs.dates) // 8)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels([hs.dates[i][:7] for i in range(0, len(hs.dates), step)], rotation=0, fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Flujos de carbono — observado vs predicho", fontsize=11)
    fig.tight_layout()
    return _png(fig)


def confusion_png(tn: int, fp: int, fn: int, tp: int, model_name: str) -> bytes:
    cm = np.array([[tn or 0, fp or 0], [fn or 0, tp or 0]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > cm.max() / 2 else _INK, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fuente", "sumidero"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fuente", "sumidero"])
    ax.set_xlabel("predicho"); ax.set_ylabel("real")
    ax.set_title(f"Matriz de confusión — {model_name}\n(holdout 2018)", fontsize=10)
    fig.tight_layout()
    return _png(fig)


def roc_png(y_true: list[int], proba: list[float], auc: float | None, model_name: str) -> bytes:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    if y_true and proba and len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, color=_EMERALD, lw=2.2,
                label=f"AUC = {auc:.3f}" if auc is not None else "ROC")
    ax.plot([0, 1], [0, 1], color="#a1a1aa", lw=1, ls="--")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos\n(recall de sumidero)")
    ax.set_title(f"Curva ROC — {model_name}", fontsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _png(fig)
