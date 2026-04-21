from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba, calc_source_with_numba_traj
from .source_rvdpisodr import SourceRvdpisodr, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier


class SourceRvdpperpdr(SourceRvdpisodr):
    def __init__(self):
        SourceRvdpisodr.__init__(self)

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho, traj=False, **kwarg
    ) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho)

    def calc_fourier(self, rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho, traj=traj)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pgyr"]


def load():
    return SourceRvdpperpdr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvdpperpdr().expr
