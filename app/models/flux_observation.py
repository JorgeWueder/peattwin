"""Serie temporal de observaciones de flujo de una turbera."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.enums import CarbonBalanceClass
from app.models.types import carbon_balance_class_enum

if TYPE_CHECKING:
    from app.models.site import Site


class FluxObservation(Base):
    __tablename__ = "flux_observations"
    __table_args__ = (
        UniqueConstraint("site_id", "timestamp", name="uq_flux_obs_site_timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(
        ForeignKey("sites.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    nee_co2: Mapped[float | None] = mapped_column(Float(), nullable=True)
    fch4: Mapped[float | None] = mapped_column(Float(), nullable=True)
    water_table_depth: Mapped[float | None] = mapped_column(Float(), nullable=True)
    soil_temp: Mapped[float | None] = mapped_column(Float(), nullable=True)
    soil_water_content: Mapped[float | None] = mapped_column(Float(), nullable=True)
    air_temp: Mapped[float | None] = mapped_column(Float(), nullable=True)
    precipitation: Mapped[float | None] = mapped_column(Float(), nullable=True)
    carbon_balance_class: Mapped[CarbonBalanceClass | None] = mapped_column(
        carbon_balance_class_enum, nullable=True
    )
    quality_flag: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    site: Mapped["Site"] = relationship(back_populates="flux_observations")
