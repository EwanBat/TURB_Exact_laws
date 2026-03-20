from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drduisodv import FluxDrduisodv, calc_in_point_with_sympy, calc_with_fourier

class FluxDrdupoldv(FluxDrduisodv):
    def __init__(self):
        FluxDrduisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, upol, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, upol, vx, vy, vz)
    
    def calc_fourier(self, rho, upol, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, upol, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','upol', 'v']

def load():
    return FluxDrdupoldv()

def print_expr():
    return FluxDrdupoldv().print_expr()