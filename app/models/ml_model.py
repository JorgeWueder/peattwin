"""Modelo de ML entrenado y registrado."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import AlgorithmType, MLTask
from app.models.types import algorithm_type_enum, ml_task_enum

if TYPE_CHECKING:
    from app.models.ml_model_metric import MLModelMetric
    from app.models.prediction import Prediction


class MLModel(Base, TimestampMixin):
    __tablename__ = "ml_models"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_ml_model_name_version"),
        # Como maximo un modelo activo por tarea.
        Index(
            "uq_ml_model_active_per_task",
            "task",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    algorithm_type: Mapped[AlgorithmType] = mapped_column(
        algorithm_type_enum, nullable=False
    )
    task: Mapped[MLTask] = mapped_column(ml_task_enum, nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    framework: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_variable: Mapped[str | None] = mapped_column(String(128), nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean(), server_default=text("false"), nullable=False
    )

    metrics: Mapped[list["MLModelMetric"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
