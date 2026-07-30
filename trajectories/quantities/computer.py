from .base import TrajectoryQuantitiesBase
from .gradient import GradientQuantitiesMixIn
from .io import QuantitiesIOMixIn


class TrajectoryQuantitiesComputer(
    GradientQuantitiesMixIn,
    QuantitiesIOMixIn,
    TrajectoryQuantitiesBase,
):
    pass
