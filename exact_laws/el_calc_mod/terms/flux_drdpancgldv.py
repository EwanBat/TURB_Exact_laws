from typing import List

from .abstract_term import calc_flux_with_numba, calc_flux_with_numba_traj
from .flux_drdpandv import FluxDrdpandv, calc_in_point_with_sympy, calc_with_fourier, calc_in_point_with_sympy_traj

class FluxDrdpancgldv(FluxDrdpandv):
    def __init__(self):
        FluxDrdpandv.__init__(self)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz)
        else:
            return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz)

    def calc_fourier(self, rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, pperpcgl, pparcgl, vx, vy, vz, pm, bx, by, bz, traj=traj)
    
    def variables(self) -> List[str]:
        return ['rho','pcgl', 'pm', 'v', 'b']

def load():
    return FluxDrdpancgldv()

def print_expr():
    return FluxDrdpandv().print_expr()