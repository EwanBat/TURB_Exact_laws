from typing import List
from numba import njit
from .abstract_term import AbstractTerm, calc_flux_with_numba


class FluxTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz, v2, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, v2)

    def variables(self) -> List[str]:
        return ['v', 'v2']


def load():
    return FluxTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, v2):
    f1 =  vx[i,j,k] * v2[ip,jp,kp] - vx[ip,jp,kp] * v2[i,j,k] 
    f2 =  vy[i,j,k] * v2[ip,jp,kp] - vy[ip,jp,kp] * v2[i,j,k] 
    f3 =  vz[i,j,k] * v2[ip,jp,kp] - vz[ip,jp,kp] * v2[i,j,k] 
    return f1, f2, f3
