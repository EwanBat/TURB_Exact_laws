from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drdpisodv import FluxDrdpisodv, calc_in_point_with_sympy, calc_with_fourier

class FluxDrdpperpdv(FluxDrdpisodv):
    def __init__(self):
        FluxDrdpisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperp, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperp, vx, vy, vz)

    def calc_fourier(self, rho, pperp, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, pperp, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','pgyr', 'v']

def load():
    return FluxDrdpperpdv()

def print_expr():
    return FluxDrdpperpdv().print_expr()