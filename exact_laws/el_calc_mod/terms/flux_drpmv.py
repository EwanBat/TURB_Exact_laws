from typing import List

from .abstract_term import calc_flux_with_numba, calc_flux_with_numba_traj
from .flux_drpisov import FluxDrpisov, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier

class FluxDrpmv(FluxDrpisov):
    def __init__(self):
        FluxDrpisov.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pm, vx, vy, vz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, pm, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pm, vx, vy, vz)

    def calc_fourier(self, rho, pm, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, pm, vx, vy, vz, traj=traj)

    def variables(self) -> List[str]:
        return ['rho','pm', 'v']

def load():
    return FluxDrpmv()

def print_expr():
    return FluxDrpmv().print_expr()