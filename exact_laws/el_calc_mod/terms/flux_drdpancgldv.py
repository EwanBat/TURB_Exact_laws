from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drdpandv import FluxDrdpandv, calc_in_point_with_sympy, calc_with_fourier

class FluxDrdpancgldv(FluxDrdpandv):
    def __init__(self):
        FluxDrdpandv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz)
    
    def calc_fourier(self, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz, **kwarg) -> List:
        return calc_with_fourier(rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz)
    
    def variables(self) -> List[str]:
        return ['rho','pcgl', 'pm', 'v', 'b']

def load():
    return FluxDrdpancgldv()

def print_expr():
    return FluxDrdpandv().print_expr()