from typing import List
from numba import njit
from .abstract_term import AbstractTerm, calc_flux_with_numba


class FluxTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['v']


def load():
    return FluxTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    f1 =  vx[i,j,k] * (vx[ip,jp,kp] **2 + vy[ip,jp,kp] **2 + vz[ip,jp,kp] **2 ) - vx[ip,jp,kp] * (vx[i,j,k] **2 + vy[i,j,k] **2 + vz[i,j,k] **2 )
    f2 =  vy[i,j,k] * (vx[ip,jp,kp] **2 + vy[ip,jp,kp] **2 + vz[ip,jp,kp] **2 ) - vy[ip,jp,kp] * (vx[i,j,k] **2 + vy[i,j,k] **2 + vz[i,j,k] **2 )
    f3 =  vz[i,j,k]  * (vx[ip,jp,kp] **2 + vy[ip,jp,kp] **2 + vz[ip,jp,kp] **2 ) - vz[ip,jp,kp] * (vx[i,j,k] **2 + vy[i,j,k] **2 + vz[i,j,k] **2 )
    return f1, f2, f3
