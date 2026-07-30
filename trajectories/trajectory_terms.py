"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.

This file is now a backward-compatibility shim.
Implementation has been split into trajectories/terms/ package.
"""

from .terms import TrajectoryTermsComputer


def compute_all_terms_for_laws(dic_quantities: dict = None, grid_param: dict = None,
                                traj_param: dict = None, physical_param: dict = None,
                                run_params: dict = None, filename: str = "terms_trajectory.h5",
                                laws: list = None, verbose: bool = False):
    computer = TrajectoryTermsComputer(verbose=verbose,
                                      physical_param=physical_param,
                                      traj_param=traj_param,
                                      grid_param=grid_param,
                                      run_params=run_params)
    return computer.compute_all_terms_for_laws(dic_quantities, laws, filename)
