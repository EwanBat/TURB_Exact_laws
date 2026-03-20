from typing import List
from numba import njit
import sympy as sp

from .abstract_term import calc_source_with_numba
from .source_ss21 import SourceSs21, calc_in_point_with_sympy

class SourceSs21Pol(SourceSs21):
    def __init__(self):
        SourceSs21.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int],vx, vy, vz, bx, by, bz, rho, pm, ppol, upol, divv, divb, dxupol, dyupol, dzupol, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, bx, by, bz, rho, pm, ppol, upol, divv, divb, dxupol, dyupol, dzupol)

    def variables(self) -> List[str]:
        return ['v', 'b', 'rho', 'pm', 'ppol', 'upol', 'divv', 'divb', 'gradupol']

def load():
    return SourceSs21Pol()

def print_expr():
    return SourceSs21Pol().print_expr()