from .base import TrajectoryLawsBase
from .coefficients import LawCoefficientsMixIn
from .io import LawsIOMixIn


class TrajectoryLawsComputer(
    LawCoefficientsMixIn,
    LawsIOMixIn,
    TrajectoryLawsBase,
):
    pass
