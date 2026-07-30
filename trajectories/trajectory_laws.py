"""
Compute law terms with coefficients along trajectories using fully vectorized operations.
Applies law coefficients to computed terms and handles divergence calculations.

This file is now a backward-compatibility shim.
Implementation has been split into trajectories/laws/ package.
"""

from .laws import TrajectoryLawsComputer


def compute_laws_terms_with_coefficients(dic_terms, physical_param=None, traj_param=None,
                                        filename="laws_terms.h5",
                                        laws=None, method: str = None,
                                        verbose: bool = False):
    computer = TrajectoryLawsComputer(verbose=verbose,
                                     physical_param=physical_param,
                                     traj_param=traj_param)
    return computer.compute_laws_terms(dic_terms, laws=laws, filename=filename, method=method)
