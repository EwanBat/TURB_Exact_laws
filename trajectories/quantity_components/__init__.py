"""
Module to compute quantities along a trajectory.
Analog to trajectory_terms (terms_components) but for quantities, using QUANTITIES objects.

Architecture: class split across two modules:
  - quantity_components/base.py   : MockFile, constants, __init__, core helper, I/O
  - quantity_components/compute.py: public entry, requirement listing, dispatch, computation
"""
from trajectories.quantity_components.base import MockFile, TrajectoryQuantitiesComputerBase
from trajectories.quantity_components.compute import TrajectoryQuantitiesComputeMixin


class TrajectoryQuantitiesComputer(
    TrajectoryQuantitiesComputerBase,
    TrajectoryQuantitiesComputeMixin,
):
    """
    Compute quantities along trajectories.

    Handles single-satellite, 4-satellite formation, and 9-satellite cube
    configurations. Manages quantity dependencies and vectorized computations.
    """


__all__ = ["TrajectoryQuantitiesComputer", "MockFile"]