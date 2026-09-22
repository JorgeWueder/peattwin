"""Entrenamiento de modelos (esqueleto ejecutable).

Ejecutar desde la raiz del repositorio, con el venv del modulo ML activo:

    python -m ml.src.train
    python -m ml.src.train --output ml/models/baseline.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ml.src.config import MODELS_DIR, RANDOM_SEED
from ml.src.data import make_synthetic_dataset

TARGET = "net_co2_flux"


def train(output: Path) -> None:
    df = make_synthetic_dataset()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE : {mean_absolute_error(y_test, preds):.4f}")
    print(f"R^2 : {r2_score(y_test, preds):.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Modelo guardado en: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo baseline de PeatTwin (esqueleto)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "baseline.joblib",
        help="Ruta de salida del modelo entrenado.",
    )
    args = parser.parse_args()
    train(args.output)


if __name__ == "__main__":
    main()
