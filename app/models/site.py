"""Sitio de turbera monitorizado."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import PeatlandType
from app.models.types import peatland_type_enum

if TYPE_CHECKING:
    from app.models.flux_observation import FluxObservation
    from app.models.restoration_scenario import RestorationScenario


class Site(Base, TimestampMixin):
    __tablename__ = "sites"
    __table_args__ = (
        UniqueConstraint("fluxnet_id", name="uq_sites_fluxnet_id"),
        UniqueConstraint("icos_id", name="uq_sites_icos_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float(), nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float(), nullable=True)
    area_ha: Mapped[float | None] = mapped_column(Float(), nullable=True)
    fluxnet_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    icos_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    peatland_type: Mapped[PeatlandType | None] = mapped_column(
        peatland_type_enum, nullable=True
    )

    flux_observations: Mapped[list["FluxObservation"]] = relationship(
        back_populates="site",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    restoration_scenarios: Mapped[list["RestorationScenario"]] = relationship(
        back_populates="site",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
