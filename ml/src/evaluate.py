"""Evaluacion de modelos entrenados (esqueleto).

    python -m ml.src.evaluate ml/models/baseline.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from ml.src.data import make_synthetic_dataset

TARGET = "net_co2_flux"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua un modelo entrenado de PeatTwin (esqueleto)."
    )
    parser.add_argument("model", type=Path, help="Ruta al archivo .joblib del modelo.")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = make_synthetic_dataset()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"R^2 sobre el dataset sintetico completo: {model.score(X, y):.4f}")


if __name__ == "__main__":
    main()
