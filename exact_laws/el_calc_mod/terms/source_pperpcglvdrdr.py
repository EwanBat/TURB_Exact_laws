from typing import List
from numba import njit
import sympy as sp
from .abstract_term import calc_source_with_numba, calc_source_with_numba_traj
from .source_pisovdrdr import SourcePisovdrdr, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier


class SourcePperpcglvdrdr(SourcePisovdrdr):
    def __init__(self):
        SourcePisovdrdr.__init__(self)

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho, traj=False, **kwarg
    ) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho)

    def calc_fourier(self, rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho, traj=traj)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pcgl"]


def load():
    return SourcePperpcglvdrdr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourcePperpcglvdrdr().expr

