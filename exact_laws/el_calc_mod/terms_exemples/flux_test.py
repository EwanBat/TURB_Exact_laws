from typing import List
from numba import njit
from .abstract_term import AbstractTerm, calc_flux_with_numba


class FluxTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz, rho, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, rho)

    def variables(self) -> List[str]:
        return ['v', 'rho']


def load():
    return FluxTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, rho):
    f1 =  rho[ip, jp, kp] * vx[i,j,k] -  rho[i, j, k] * vx[ip,jp,kp]
    f2 =  rho[ip, jp, kp] * vy[i,j,k] - rho[i, j, k] * vy[ip,jp,kp] 
    f3 =  rho[ip, jp, kp] * vz[i,j,k] - rho[i, j, k] * vz[ip,jp,kp] 
    return f1, f2, f3
