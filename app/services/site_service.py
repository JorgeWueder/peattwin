"""Capa de servicio para el recurso Site (CRUD)."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.site import Site
from app.schemas.site import SiteCreate, SiteUpdate


def list_sites(db: Session, skip: int = 0, limit: int = 100) -> list[Site]:
    return list(
        db.scalars(select(Site).order_by(Site.id).offset(skip).limit(limit))
    )


def get_site(db: Session, site_id: int) -> Site | None:
    return db.get(Site, site_id)


def create_site(db: Session, payload: SiteCreate) -> Site:
    site = Site(**payload.model_dump())
    db.add(site)
    db.commit()
    db.refresh(site)
    return site


def update_site(db: Session, site_id: int, payload: SiteUpdate) -> Site | None:
    site = db.get(Site, site_id)
    if site is None:
        return None
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(site, key, value)
    db.commit()
    db.refresh(site)
    return site


def delete_site(db: Session, site_id: int) -> bool:
    site = db.get(Site, site_id)
    if site is None:
        return False
    db.delete(site)
    db.commit()
    return True
