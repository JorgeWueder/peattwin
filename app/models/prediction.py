"""Predicciones del modelo activo sobre un escenario de restauracion."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.enums import CarbonBalanceClass
from app.models.types import carbon_balance_class_enum

if TYPE_CHECKING:
    from app.models.ml_model import MLModel
    from app.models.restoration_scenario import RestorationScenario


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "scenario_id",
            "model_id",
            "timestamp",
            name="uq_prediction_scenario_model_ts",
        ),
        Index("ix_predictions_model_id", "model_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("restoration_scenarios.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    co2_predicted: Mapped[float | None] = mapped_column(Float(), nullable=True)
    ch4_predicted: Mapped[float | None] = mapped_column(Float(), nullable=True)
    predicted_class: Mapped[CarbonBalanceClass | None] = mapped_column(
        carbon_balance_class_enum, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scenario: Mapped["RestorationScenario"] = relationship(back_populates="predictions")
    model: Mapped["MLModel"] = relationship(back_populates="predictions")
