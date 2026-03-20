from typing import List
from numba import njit
import sympy as sp

from .abstract_term import calc_flux_with_numba
from .flux_ss21hyb import FluxSs21Hyb, calc_in_point_with_sympy

class FluxSs21HybIso(FluxSs21Hyb):
    def __init__(self):
        FluxSs21Hyb.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, rho, pm, piso, uiso, **kwarg) -> List[float]:
        #return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, rho, pm, piso, uiso)

    def variables(self) -> List[str]:
        return ['v', 'rho', 'pm', 'piso', 'uiso']

def load():
    return FluxSs21HybIso()

def print_expr():
    return FluxSs21HybIso().print_expr()