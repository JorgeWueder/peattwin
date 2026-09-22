"""Esquemas Pydantic para el recurso Site (sitio de turbera)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import PeatlandType


class SiteBase(BaseModel):
    name: str
    description: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    area_ha: float | None = Field(default=None, ge=0)
    fluxnet_id: str | None = Field(default=None, max_length=32)
    icos_id: str | None = Field(default=None, max_length=32)
    peatland_type: PeatlandType | None = None


class SiteCreate(SiteBase):
    pass


class SiteUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    area_ha: float | None = Field(default=None, ge=0)
    fluxnet_id: str | None = Field(default=None, max_length=32)
    icos_id: str | None = Field(default=None, max_length=32)
    peatland_type: PeatlandType | None = None


class SiteRead(SiteBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
