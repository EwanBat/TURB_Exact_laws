from typing import List

from .abstract_term import calc_flux_with_numba, calc_flux_with_numba_traj
from .flux_drduisodv import FluxDrduisodv, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier

class FluxDrducgldv(FluxDrduisodv):
    def __init__(self):
        FluxDrduisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, ucgl, vx, vy, vz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, ucgl, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, ucgl, vx, vy, vz)
    
    def calc_fourier(self, rho, ucgl, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, ucgl, vx, vy, vz, traj=traj)

    def variables(self) -> List[str]:
        return ['rho','ucgl', 'v']

def load():
    return FluxDrducgldv()

def print_expr():
    return FluxDrducgldv().print_expr()