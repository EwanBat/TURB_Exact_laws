from typing import List
from numba import njit
import sympy as sp

from .abstract_term import calc_source_with_numba
from .source_ss22 import SourceSs22, calc_in_point_with_sympy

class SourceSs22Iso(SourceSs22):
    def __init__(self):
        SourceSs22.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int],vx, vy, vz, bx, by, bz, rho, pm, piso, uiso, divv, divb, dxrho, dyrho, dzrho, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, bx, by, bz, rho, pm, piso, uiso, divv, divb, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ['v', 'b', 'rho', 'pm', 'piso', 'uiso', 'divv', 'divb', 'gradrho']

def load():
    return SourceSs22Iso()

def print_expr():
    return SourceSs22Iso().print_expr()