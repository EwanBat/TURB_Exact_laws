from typing import List

from .abstract_term import calc_flux_with_numba, calc_flux_with_numba_traj
from .flux_drdpisodv import FluxDrdpisodv, calc_in_point_with_sympy, calc_with_fourier, calc_in_point_with_sympy_traj

class FluxDrdpmdv(FluxDrdpisodv):
    def __init__(self):
        FluxDrdpisodv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pm, vx, vy, vz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, pm, vx, vy, vz)
        else:
            return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pm, vx, vy, vz)

    def calc_fourier(self, rho, pm, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, pm, vx, vy, vz, traj=traj)
    
    def variables(self) -> List[str]:
        return ['rho','pm', 'v']

def load():
    return FluxDrdpmdv()

def print_expr():
    return FluxDrdpmdv().print_expr()
