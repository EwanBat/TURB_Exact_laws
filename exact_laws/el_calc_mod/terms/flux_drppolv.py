from typing import List

from .abstract_term import calc_flux_with_numba
from .flux_drpisov import FluxDrpisov, calc_in_point_with_sympy, calc_with_fourier

class FluxDrppolv(FluxDrpisov):
    def __init__(self):
        FluxDrpisov.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ppol, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, ppol, vx, vy, vz)

    def calc_fourier(self, rho, ppol, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, ppol, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','ppol', 'v']

def load():
    return FluxDrppolv()

def print_expr():
    return FluxDrppolv().print_expr()