"""Esquemas Pydantic del proyecto (re-exportados para comodidad)."""
from app.schemas.auth import (
    PermissionCreate,
    PermissionRead,
    PermissionUpdate,
    RoleCreate,
    RoleRead,
    RoleUpdate,
    UserCreate,
    UserRead,
    UserUpdate,
)
from app.schemas.flux_observation import (
    FluxObservationCreate,
    FluxObservationRead,
    FluxObservationUpdate,
)
from app.schemas.ml_model import MLModelCreate, MLModelRead, MLModelUpdate
from app.schemas.ml_model_metric import MLModelMetricCreate, MLModelMetricRead
from app.schemas.prediction import PredictionCreate, PredictionRead
from app.schemas.restoration_scenario import (
    RestorationScenarioCreate,
    RestorationScenarioRead,
    RestorationScenarioUpdate,
)
from app.schemas.site import SiteCreate, SiteRead, SiteUpdate

__all__ = [
    "PermissionCreate",
    "PermissionRead",
    "PermissionUpdate",
    "RoleCreate",
    "RoleRead",
    "RoleUpdate",
    "UserCreate",
    "UserRead",
    "UserUpdate",
    "SiteCreate",
    "SiteRead",
    "SiteUpdate",
    "FluxObservationCreate",
    "FluxObservationRead",
    "FluxObservationUpdate",
    "RestorationScenarioCreate",
    "RestorationScenarioRead",
    "RestorationScenarioUpdate",
    "MLModelCreate",
    "MLModelRead",
    "MLModelUpdate",
    "MLModelMetricCreate",
    "MLModelMetricRead",
    "PredictionCreate",
    "PredictionRead",
]
