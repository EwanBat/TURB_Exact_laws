from typing import List
from numba import njit
import sympy as sp

from .abstract_term import calc_source_with_numba
from .forc_v import ForcV, calc_in_point_with_sympy, calc_with_fourier

class DissV(ForcV):
    def __init__(self):
        ForcV.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, hdk2x, hdk2y, hdk2z, **kwarg
        ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, hdk2x, hdk2y, hdk2z)

    def calc_fourier(self, rho, vx, vy, vz, hdk2x, hdk2y, hdk2z, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, hdk2x, hdk2y, hdk2z)
    
    def variables(self) -> List[str]:
        return ["hdk2", "v", "rho"]

def load():
    return DissV()

def print_expr():
    return DissV().print_expr()


