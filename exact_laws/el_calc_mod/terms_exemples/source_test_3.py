from typing import List
from numba import njit
import numpy as np
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz,dxv2, dyv2, dzv2, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz,dxv2, dyv2, dzv2)

    def variables(self) -> List[str]:
        return ['v', 'gradv2']


def load():
    return SourceTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz,dxv2, dyv2, dzv2):
    return (vx[i, j, k] * dxv2[ip, jp, kp] + vy[i, j, k] * dyv2[ip, jp, kp] + vz[i, j, k] * dzv2[ip, jp, kp]
            + vx[ip, jp, kp] * dxv2[i, j, k] + vy[ip, jp, kp] * dyv2[i, j, k] + vz[ip, jp, kp] * dzv2[i, j, k])
