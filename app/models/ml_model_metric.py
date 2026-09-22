"""Metricas de evaluacion por modelo y por fold de validacion cruzada."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ml_model import MLModel


class MLModelMetric(Base):
    __tablename__ = "ml_model_metrics"
    __table_args__ = (
        UniqueConstraint("model_id", "fold", name="uq_ml_model_metric_model_fold"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False
    )
    # 0..k-1 para cada fold; usar -1 para el resultado agregado.
    fold: Mapped[int] = mapped_column(Integer(), nullable=False)

    # Regresion
    rmse: Mapped[float | None] = mapped_column(Float(), nullable=True)
    mae: Mapped[float | None] = mapped_column(Float(), nullable=True)
    r2: Mapped[float | None] = mapped_column(Float(), nullable=True)
    nse: Mapped[float | None] = mapped_column(Float(), nullable=True)
    kge: Mapped[float | None] = mapped_column(Float(), nullable=True)

    # Clasificacion
    accuracy: Mapped[float | None] = mapped_column(Float(), nullable=True)
    precision: Mapped[float | None] = mapped_column(Float(), nullable=True)
    recall: Mapped[float | None] = mapped_column(Float(), nullable=True)
    f1: Mapped[float | None] = mapped_column(Float(), nullable=True)
    auc_roc: Mapped[float | None] = mapped_column(Float(), nullable=True)

    training_time_seconds: Mapped[float | None] = mapped_column(Float(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model: Mapped["MLModel"] = relationship(back_populates="metrics")
