from typing import List
from numba import njit
import sympy as sp
from .abstract_term import calc_source_with_numba
from .source_pisovdrdr import SourcePisovdrdr, calc_in_point_with_sympy, calc_with_fourier


class SourcePperpvdrdr(SourcePisovdrdr):
    def __init__(self):
        SourcePisovdrdr.__init__(self)

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho)

    def calc_fourier(self, rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pperp, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pgyr"]


def load():
    return SourcePperpvdrdr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourcePperpvdrdr().expr
