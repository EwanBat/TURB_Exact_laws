from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba
from .source_rvbetadu import SourceRvbetadu, calc_in_point_with_sympy, calc_with_fourier


class SourceRvbetadupol(SourceRvbetadu):
    def __init__(self):
        SourceRvbetadu.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pm, ppol, dxupol, dyupol, dzupol, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pm, ppol, dxupol, dyupol, dzupol)

    def calc_fourier(self, rho, vx, vy, vz, pm, ppol, dxupol, dyupol, dzupol, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pm, ppol, dxupol, dyupol, dzupol)

    def variables(self) -> List[str]:
        return ["rho", "gradupol", "v", "pm", "ppol"]


def load():
    return SourceRvbetadupol()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvbetadupol().expr

