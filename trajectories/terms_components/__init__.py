"""
Module to compute terms along a trajectory.
Analog to quantity_components (trajectory_quantities) but for terms.
Uses calc_fourier() methods from terms for trajectories.
Encapsulated in TrajectoryTermsComputer class for better parameter management.

Architecture: class split across three mixin modules:
  - terms_components/base.py       : constants, __init__, helpers, dispatcher, I/O
  - terms_components/incremental.py: incremental computation methods
  - terms_components/fourier.py    : Fourier computation methods
"""
from trajectories.terms_components.base import logger, TrajectoryTermsComputerBase
from trajectories.terms_components.incremental import TrajectoryTermsIncrementalMixin
from trajectories.terms_components.fourier import TrajectoryTermsFourierMixin


class TrajectoryTermsComputer(
    TrajectoryTermsComputerBase,
    TrajectoryTermsIncrementalMixin,
    TrajectoryTermsFourierMixin,
):
    """
    Compute physics terms along trajectories.

    Encapsulates term computation logic with parameter storage as instance attributes
    to reduce repeated parameter passing.
    Handles both single satellite and 4-satellite formation configurations.
    """


__all__ = ["TrajectoryTermsComputer"]