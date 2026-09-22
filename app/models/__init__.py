"""Modelos ORM.

Se importan todos aqui para que Alembic (y el registro de mapeadores de
SQLAlchemy) los detecte al cargar `app.models`.
"""
from app.models.base import Base, TimestampMixin
from app.models.associations import role_permissions, user_roles
from app.models.enums import (
    AlgorithmType,
    CarbonBalanceClass,
    MLTask,
    PeatlandType,
    classify_carbon_balance,
)
from app.models.user import User
from app.models.role import Role
from app.models.permission import Permission
from app.models.site import Site
from app.models.flux_observation import FluxObservation
from app.models.restoration_scenario import RestorationScenario
from app.models.ml_model import MLModel
from app.models.ml_model_metric import MLModelMetric
from app.models.prediction import Prediction

__all__ = [
    "Base",
    "TimestampMixin",
    "user_roles",
    "role_permissions",
    "AlgorithmType",
    "CarbonBalanceClass",
    "MLTask",
    "PeatlandType",
    "classify_carbon_balance",
    "User",
    "Role",
    "Permission",
    "Site",
    "FluxObservation",
    "RestorationScenario",
    "MLModel",
    "MLModelMetric",
    "Prediction",
]
