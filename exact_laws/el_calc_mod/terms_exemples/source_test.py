from typing import List
from numba import njit
import numpy as np
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz,dxrho, dyrho, dzrho, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz,dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ['v', 'gradrho']


def load():
    return SourceTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz,dxrho, dyrho, dzrho):
    return (vx[i, j, k] * dxrho[ip, jp, kp] + vy[i, j, k] * dyrho[ip, jp, kp] + vz[i, j, k] * dzrho[ip, jp, kp]
            + vx[ip, jp, kp] * dxrho[i, j, k] + vy[ip, jp, kp] * dyrho[i, j, k] + vz[ip, jp, kp] * dzrho[i, j, k])
