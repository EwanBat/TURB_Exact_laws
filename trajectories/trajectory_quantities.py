"""
Compute quantities along trajectories using fully vectorized operations.
Analog to trajectory_terms.py but for quantities, using QUANTITIES objects.

This file is now a backward-compatibility shim.
Implementation has been split into trajectories/quantities/ package.
"""

from .quantities import TrajectoryQuantitiesComputer


def extract_and_compute_trajectory_quantities(dic_datas: dict, grid_param: dict = None,
                                              traj_param: dict = None, physical_param: dict = None,
                                              laws=None, terms=None, quantities=None,
                                              method: str = None, verbose: bool = False,
                                              filename: str = "computed_quantities.h5"):
    computer = TrajectoryQuantitiesComputer(
        verbose=verbose, grid_param=grid_param,
        physical_param=physical_param, traj_param=traj_param
    )
    return computer.extract_and_compute(
        dic_datas, laws=laws, terms=terms,
        quantities=quantities, method=method, filename=filename
    )
