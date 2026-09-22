"""Tipos SQLAlchemy compartidos (ENUM nativos de PostgreSQL).

Instancias unicas y reutilizables: SQLAlchemy crea cada tipo ENUM una sola vez
aunque se use en varias tablas.
"""
from sqlalchemy import Enum as SAEnum

from app.models.enums import AlgorithmType, CarbonBalanceClass, MLTask, PeatlandType

peatland_type_enum = SAEnum(PeatlandType, name="peatland_type")
carbon_balance_class_enum = SAEnum(CarbonBalanceClass, name="carbon_balance_class")
algorithm_type_enum = SAEnum(AlgorithmType, name="algorithm_type")
ml_task_enum = SAEnum(MLTask, name="ml_task")
