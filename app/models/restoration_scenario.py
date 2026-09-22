"""Escenario de restauracion hidrologica sobre un sitio."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.prediction import Prediction
    from app.models.site import Site
    from app.models.user import User


class RestorationScenario(Base, TimestampMixin):
    __tablename__ = "restoration_scenarios"
    __table_args__ = (
        UniqueConstraint(
            "site_id", "name", name="uq_restoration_scenario_site_name"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(
        ForeignKey("sites.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_water_table_depth: Mapped[float | None] = mapped_column(Float(), nullable=True)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    site: Mapped["Site"] = relationship(back_populates="restoration_scenarios")
    created_by: Mapped["User | None"] = relationship()
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="scenario",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
