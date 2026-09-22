"""CRUD de restoration_scenarios (escenarios de restauracion hidrologica)."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.restoration_scenario import RestorationScenario
from app.schemas.restoration_scenario import (
    RestorationScenarioCreate,
    RestorationScenarioUpdate,
)


def list_scenarios(
    db: Session, *, site_id: int | None = None, skip: int = 0, limit: int = 100
) -> list[RestorationScenario]:
    stmt = select(RestorationScenario).order_by(RestorationScenario.id)
    if site_id is not None:
        stmt = stmt.where(RestorationScenario.site_id == site_id)
    return list(db.scalars(stmt.offset(skip).limit(limit)))


def get_scenario(db: Session, scenario_id: int) -> RestorationScenario | None:
    return db.get(RestorationScenario, scenario_id)


def create_scenario(
    db: Session, payload: RestorationScenarioCreate, *, created_by_id: int | None
) -> RestorationScenario:
    data = payload.model_dump()
    data["created_by_id"] = created_by_id  # el autor es el usuario autenticado
    scenario = RestorationScenario(**data)
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return scenario


def update_scenario(
    db: Session, scenario_id: int, payload: RestorationScenarioUpdate
) -> RestorationScenario | None:
    scenario = db.get(RestorationScenario, scenario_id)
    if scenario is None:
        return None
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(scenario, key, value)
    db.commit()
    db.refresh(scenario)
    return scenario


def delete_scenario(db: Session, scenario_id: int) -> bool:
    scenario = db.get(RestorationScenario, scenario_id)
    if scenario is None:
        return False
    db.delete(scenario)
    db.commit()
    return True
