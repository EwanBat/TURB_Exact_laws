# trajectory_terms.py
"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.
Encapsulated in TrajectoryTermsComputer class for better parameter management.

Architecture: class split across three mixin modules:
  - terms_components/base.py       : constants, __init__, helpers, dispatcher, I/O
  - terms_components/incremental.py: incremental computation methods
  - terms_components/fourier.py    : Fourier computation methods
"""
import logging

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


# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def compute_all_terms_for_laws(dic_quantities: dict = None, grid_param: dict = None, traj_param: dict = None, physical_param: dict = None, run_params: dict = None, filename:str = "terms_trajectory.h5", laws: list = None, verbose: bool = False):
    """
    Backward compatibility wrapper for compute_all_terms_for_laws.

    Deprecated: Use TrajectoryTermsComputer class instead.
    """
    computer = TrajectoryTermsComputer(verbose=verbose,
                                      physical_param=physical_param,
                                      traj_param=traj_param,
                                      grid_param=grid_param,
                                      run_params=run_params)
    return computer.compute_all_terms_for_laws(dic_quantities, laws, filename)
