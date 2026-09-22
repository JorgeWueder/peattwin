"""Enumeraciones del dominio del gemelo digital."""
import enum


class PeatlandType(str, enum.Enum):
    """Tipo de turbera."""

    BOG = "BOG"
    FEN = "FEN"
    BLANKET_BOG = "BLANKET_BOG"
    RAISED_BOG = "RAISED_BOG"
    TRANSITIONAL_MIRE = "TRANSITIONAL_MIRE"
    SWAMP_FOREST = "SWAMP_FOREST"
    OTHER = "OTHER"


class CarbonBalanceClass(str, enum.Enum):
    """Clase de balance de carbono derivada del signo del NEE."""

    SINK = "SINK"        # NEE < 0  -> la turbera absorbe carbono
    SOURCE = "SOURCE"    # NEE > 0  -> la turbera emite carbono
    NEUTRAL = "NEUTRAL"  # NEE ~ 0


class AlgorithmType(str, enum.Enum):
    """Familia del algoritmo del modelo."""

    CLASSICAL = "CLASSICAL"  # clasico (p.ej. Random Forest, XGBoost)
    HYBRID = "HYBRID"        # hibrido (p.ej. proceso + red neuronal)


class MLTask(str, enum.Enum):
    """Tarea de aprendizaje automatico."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"


def classify_carbon_balance(
    nee: float | None, neutral_band: float = 0.0
) -> CarbonBalanceClass | None:
    """Deriva la clase de balance de carbono a partir del signo del NEE.

    - NEE < -neutral_band  -> SINK (sumidero)
    - NEE >  neutral_band  -> SOURCE (fuente)
    - en otro caso         -> NEUTRAL
    """
    if nee is None:
        return None
    if nee < -neutral_band:
        return CarbonBalanceClass.SINK
    if nee > neutral_band:
        return CarbonBalanceClass.SOURCE
    return CarbonBalanceClass.NEUTRAL
