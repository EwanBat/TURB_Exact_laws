"""
Module to compute law terms with coefficients along a trajectory.
Analog to terms_components and quantity_components but for laws.

Architecture: class split across two modules:
  - laws_components/base.py        : logger, __init__, parameter helper, I/O
  - laws_components/coefficients.py: public entry, dispatch, coefficient application
"""
from trajectories.laws_components.base import TrajectoryLawsComputerBase
from trajectories.laws_components.coefficients import TrajectoryLawsCoefficientsMixin


class TrajectoryLawsComputer(
    TrajectoryLawsComputerBase,
    TrajectoryLawsCoefficientsMixin,
):
    """
    Compute law terms with coefficients along trajectories.

    Applies law coefficients to computed terms, handles divergence calculations,
    and manages both single-satellite and 4-satellite configurations.
    All data maintains structure: {sat_name: {term_name: array(n_traj, n_pts)}}
    """


__all__ = ["TrajectoryLawsComputer"]