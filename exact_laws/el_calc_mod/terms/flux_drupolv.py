from typing import List

from .abstract_term import calc_flux_with_numba, calc_flux_with_numba_traj
from .flux_druisov import FluxDruisov, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier

class FluxDrupolv(FluxDruisov):
    def __init__(self):
        FluxDruisov.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, upol, vx, vy, vz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, upol, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, upol, vx, vy, vz)

    def calc_fourier(self, rho, upol, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, upol, vx, vy, vz, traj=traj)

    def variables(self) -> List[str]:
        return ['rho','upol', 'v']

def load():
    return FluxDrupolv()

def print_expr():
    return FluxDrupolv().print_expr()