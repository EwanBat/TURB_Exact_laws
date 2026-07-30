from .base import TrajectoryTermsBase
from .incremental import IncrementalTermsMixIn
from .fourier import FourierTermsMixIn
from .io import TermsIOMixIn


class TrajectoryTermsComputer(
    IncrementalTermsMixIn,
    FourierTermsMixIn,
    TermsIOMixIn,
    TrajectoryTermsBase,
):
    pass
