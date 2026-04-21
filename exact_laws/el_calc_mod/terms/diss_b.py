from typing import List
import sympy as sp

from .abstract_term import calc_source_with_numba, calc_source_with_numba_traj
from .forc_v import ForcV, calc_in_point_with_sympy, calc_with_fourier, calc_in_point_with_sympy_traj

class DissB(ForcV):
    def __init__(self):
        ForcV.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, hdmx, hdmy, hdmz, traj=False, **kwarg
        ) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, bx, by, bz, hdmx, hdmy, hdmz)
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, hdmx, hdmy, hdmz)

    def calc_fourier(self, rho, bx, by, bz, hdmx, hdmy, hdmz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, bx, by, bz, hdmx, hdmy, hdmz, traj=traj)
    
    def variables(self) -> List[str]:
        return ["hdm", "b", "rho"]

def load():
    return DissB()

def print_expr():
    return DissB().print_expr()
