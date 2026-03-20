from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drdpisodv import FluxDrdpisodv, calc_in_point_with_sympy, calc_with_fourier

class FluxDrdpperpcgldv(FluxDrdpisodv):
    def __init__(self):
        FluxDrdpisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperpcgl, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperpcgl, vx, vy, vz)

    def calc_fourier(self, rho, pperpcgl, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, pperpcgl, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pcgl', 'v']

def load():
    return FluxDrdpperpcgldv()

def print_expr():
    return FluxDrdpperpcgldv().print_expr()