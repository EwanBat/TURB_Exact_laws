from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba, calc_source_with_numba_traj
from .source_rduisodv import SourceRduisodv, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier


class SourceRppoldv(SourceRduisodv):
    def __init__(self):
        SourceRduisodv.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, ppol, divv, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, ppol, divv)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, ppol, divv)

    def calc_fourier(self, rho, ppol, divv, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, ppol, divv, traj=traj)

    def variables(self) -> List[str]:
        return ["rho", "ppol", "divv"]


def load():
    return SourceRppoldv()


def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRppoldv().expr
