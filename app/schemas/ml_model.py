"""Esquemas Pydantic para los modelos de ML registrados."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import AlgorithmType, MLTask


class MLModelBase(BaseModel):
    name: str
    description: str | None = None
    algorithm_type: AlgorithmType
    task: MLTask
    version: str
    file_path: str
    framework: str | None = None
    target_variable: str | None = None
    trained_at: datetime | None = None
    is_active: bool = False


class MLModelCreate(MLModelBase):
    pass


class MLModelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    algorithm_type: AlgorithmType | None = None
    task: MLTask | None = None
    version: str | None = None
    file_path: str | None = None
    framework: str | None = None
    target_variable: str | None = None
    trained_at: datetime | None = None
    is_active: bool | None = None


class MLModelRead(MLModelBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
