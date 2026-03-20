from typing import List
from numba import njit
import sympy as sp

from .abstract_term import calc_flux_with_numba
from .flux_ss22 import FluxSs22, calc_in_point_with_sympy

class FluxSs22Pol(FluxSs22):
    def __init__(self):
        FluxSs22.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, bx, by, bz, rho, pm, ppol, upol, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, bx, by, bz, rho, pm, ppol, upol)

    def variables(self) -> List[str]:
        return ['v', 'b', 'rho', 'pm', 'ppol', 'upol']

def load():
    return FluxSs22Pol()

def print_expr():
    return FluxSs22Pol().print_expr()

