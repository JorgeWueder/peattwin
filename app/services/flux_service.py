"""Consulta de flux_observations con filtros por sitio y rango de fechas."""
from __future__ import annotations

from datetime import date, datetime, time, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import CarbonBalanceClass
from app.models.flux_observation import FluxObservation
from app.schemas.flux_observation import FluxObservationCreate


def query(
    db: Session,
    *,
    site_id: int | None = None,
    start: date | None = None,
    end: date | None = None,
    cls: CarbonBalanceClass | None = None,
    skip: int = 0,
    limit: int = 200,
) -> list[FluxObservation]:
    stmt = select(FluxObservation).order_by(
        FluxObservation.site_id, FluxObservation.timestamp
    )
    if site_id is not None:
        stmt = stmt.where(FluxObservation.site_id == site_id)
    if start is not None:
        lo = datetime.combine(start, time.min, tzinfo=timezone.utc)
        stmt = stmt.where(FluxObservation.timestamp >= lo)
    if end is not None:
        hi = datetime.combine(end, time.max, tzinfo=timezone.utc)
        stmt = stmt.where(FluxObservation.timestamp <= hi)
    if cls is not None:
        stmt = stmt.where(FluxObservation.carbon_balance_class == cls)
    stmt = stmt.offset(skip).limit(limit)
    return list(db.scalars(stmt))


def create(db: Session, payload: FluxObservationCreate) -> FluxObservation:
    obs = FluxObservation(**payload.model_dump())
    db.add(obs)
    db.commit()
    db.refresh(obs)
    return obs
