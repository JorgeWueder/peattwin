"""digital twin schema: turberas, flujos, escenarios, ML y auth

Revision ID: 0002_digital_twin_schema
Revises: 0001_initial
Create Date: 2026-08-31

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_digital_twin_schema"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PEATLAND_TYPE_VALUES = (
    "BOG",
    "FEN",
    "BLANKET_BOG",
    "RAISED_BOG",
    "TRANSITIONAL_MIRE",
    "SWAMP_FOREST",
    "OTHER",
)
CARBON_BALANCE_VALUES = ("SINK", "SOURCE", "NEUTRAL")
ALGORITHM_TYPE_VALUES = ("CLASSICAL", "HYBRID")
ML_TASK_VALUES = ("REGRESSION", "CLASSIFICATION")

_ENUMS = (
    (PEATLAND_TYPE_VALUES, "peatland_type"),
    (CARBON_BALANCE_VALUES, "carbon_balance_class"),
    (ALGORITHM_TYPE_VALUES, "algorithm_type"),
    (ML_TASK_VALUES, "ml_task"),
)


def _enum(name: str) -> postgresql.ENUM:
    """Referencia a un tipo ENUM ya existente (no lo (re)crea)."""
    values = next(v for v, n in _ENUMS if n == name)
    return postgresql.ENUM(*values, name=name, create_type=False)


def upgrade() -> None:
    bind = op.get_bind()

    # ------------------------------------------------------------------ ENUMs
    for values, name in _ENUMS:
        postgresql.ENUM(*values, name=name).create(bind, checkfirst=True)

    # ---------------------------------------------------------- ampliar sites
    op.add_column("sites", sa.Column("fluxnet_id", sa.String(length=32), nullable=True))
    op.add_column("sites", sa.Column("icos_id", sa.String(length=32), nullable=True))
    op.add_column(
        "sites",
        sa.Column("peatland_type", _enum("peatland_type"), nullable=True),
    )
    op.create_unique_constraint("uq_sites_fluxnet_id", "sites", ["fluxnet_id"])
    op.create_unique_constraint("uq_sites_icos_id", "sites", ["icos_id"])

    # -------------------------------------------------------------------- auth
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("username", sa.String(length=150), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column(
            "is_superuser",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_roles_name", "roles", ["name"], unique=True)

    op.create_table(
        "permissions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_permissions_code", "permissions", ["code"], unique=True)

    op.create_table(
        "user_roles",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "role_id"),
    )
    op.create_table(
        "role_permissions",
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("permission_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["permission_id"], ["permissions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("role_id", "permission_id"),
    )

    # ---------------------------------------------------- dominio gemelo digital
    op.create_table(
        "flux_observations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("site_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("nee_co2", sa.Float(), nullable=True),
        sa.Column("fch4", sa.Float(), nullable=True),
        sa.Column("water_table_depth", sa.Float(), nullable=True),
        sa.Column("soil_temp", sa.Float(), nullable=True),
        sa.Column("soil_water_content", sa.Float(), nullable=True),
        sa.Column("air_temp", sa.Float(), nullable=True),
        sa.Column("precipitation", sa.Float(), nullable=True),
        sa.Column(
            "carbon_balance_class",
            _enum("carbon_balance_class"),
            nullable=True,
        ),
        sa.Column("quality_flag", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id", "timestamp", name="uq_flux_obs_site_timestamp"
        ),
    )

    op.create_table(
        "restoration_scenarios",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("site_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("target_water_table_depth", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id", "name", name="uq_restoration_scenario_site_name"
        ),
    )

    op.create_table(
        "ml_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("algorithm_type", _enum("algorithm_type"), nullable=False),
        sa.Column("task", _enum("ml_task"), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("framework", sa.String(length=64), nullable=True),
        sa.Column("target_variable", sa.String(length=128), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_ml_model_name_version"),
    )
    op.create_index("ix_ml_models_name", "ml_models", ["name"])
    op.create_index(
        "uq_ml_model_active_per_task",
        "ml_models",
        ["task"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "ml_model_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("fold", sa.Integer(), nullable=False),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("r2", sa.Float(), nullable=True),
        sa.Column("nse", sa.Float(), nullable=True),
        sa.Column("kge", sa.Float(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("auc_roc", sa.Float(), nullable=True),
        sa.Column("training_time_seconds", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_id", "fold", name="uq_ml_model_metric_model_fold"
        ),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("scenario_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("co2_predicted", sa.Float(), nullable=True),
        sa.Column("ch4_predicted", sa.Float(), nullable=True),
        sa.Column("predicted_class", _enum("carbon_balance_class"), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["scenario_id"], ["restoration_scenarios.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["model_id"], ["ml_models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scenario_id",
            "model_id",
            "timestamp",
            name="uq_prediction_scenario_model_ts",
        ),
    )
    op.create_index("ix_predictions_model_id", "predictions", ["model_id"])

    # ------------------------------------------------ datos semilla (roles/perms)
    roles_tbl = sa.table(
        "roles", sa.column("name", sa.String), sa.column("description", sa.String)
    )
    op.bulk_insert(
        roles_tbl,
        [
            {"name": "admin", "description": "Acceso total al sistema"},
            {
                "name": "investigador",
                "description": "Gestiona sitios, escenarios y modelos; ejecuta predicciones",
            },
            {
                "name": "visualizador",
                "description": "Acceso de solo lectura a datos y resultados",
            },
        ],
    )

    perms_tbl = sa.table(
        "permissions",
        sa.column("code", sa.String),
        sa.column("description", sa.String),
    )
    op.bulk_insert(
        perms_tbl,
        [
            {"code": "sites:read", "description": "Ver sitios de turbera"},
            {"code": "sites:write", "description": "Crear y editar sitios de turbera"},
            {"code": "flux:read", "description": "Ver observaciones de flujo"},
            {"code": "flux:write", "description": "Cargar observaciones de flujo"},
            {"code": "scenarios:read", "description": "Ver escenarios de restauracion"},
            {
                "code": "scenarios:write",
                "description": "Crear y editar escenarios de restauracion",
            },
            {"code": "models:read", "description": "Ver modelos de ML y sus metricas"},
            {"code": "models:write", "description": "Registrar y activar modelos de ML"},
            {"code": "predictions:read", "description": "Ver predicciones"},
            {
                "code": "predictions:run",
                "description": "Ejecutar predicciones con el modelo activo",
            },
            {
                "code": "users:manage",
                "description": "Administrar usuarios, roles y permisos",
            },
        ],
    )

    op.execute(
        """
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id FROM roles r CROSS JOIN permissions p
        WHERE r.name = 'admin'
        """
    )
    op.execute(
        """
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id FROM roles r CROSS JOIN permissions p
        WHERE r.name = 'investigador' AND p.code <> 'users:manage'
        """
    )
    op.execute(
        """
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id FROM roles r CROSS JOIN permissions p
        WHERE r.name = 'visualizador' AND split_part(p.code, ':', 2) = 'read'
        """
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("ml_model_metrics")
    op.drop_table("ml_models")
    op.drop_table("restoration_scenarios")
    op.drop_table("flux_observations")
    op.drop_table("role_permissions")
    op.drop_table("user_roles")
    op.drop_table("permissions")
    op.drop_table("roles")
    op.drop_table("users")

    op.drop_constraint("uq_sites_icos_id", "sites", type_="unique")
    op.drop_constraint("uq_sites_fluxnet_id", "sites", type_="unique")
    op.drop_column("sites", "peatland_type")
    op.drop_column("sites", "icos_id")
    op.drop_column("sites", "fluxnet_id")

    bind = op.get_bind()
    for _values, name in reversed(_ENUMS):
        postgresql.ENUM(name=name).drop(bind, checkfirst=True)
