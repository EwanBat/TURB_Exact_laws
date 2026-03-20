from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba
from .source_rvdpisodr import SourceRvdpisodr, calc_in_point_with_sympy, calc_with_fourier


class SourceRvdpperpcgldr(SourceRvdpisodr):
    def __init__(self):
        SourceRvdpisodr.__init__(self)

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pperpcgl, dxrho, dyrho, dzrho)

    def calc_fourier(self, rho, vx, vy, vz, ppperpcgl, dxrho, dyrho, dzrho, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, ppperpcgl, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pcgl"]


def load():
    return SourceRvdpperpcgldr()


def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvdpperpcgldr().expr