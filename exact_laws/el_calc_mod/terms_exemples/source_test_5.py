from typing import List
from numba import njit
import numpy as np
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], vx, vy, vz,dxuiso, dyuiso, dzuiso, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz,dxuiso, dyuiso, dzuiso)

    def variables(self) -> List[str]:
        return ['v', 'graduiso']


def load():
    return SourceTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz,dxuiso, dyuiso, dzuiso):
    return (vx[i, j, k] * dxuiso[ip, jp, kp] + vy[i, j, k] * dyuiso[ip, jp, kp] + vz[i, j, k] * dzuiso[ip, jp, kp]
            + vx[ip, jp, kp] * dxuiso[i, j, k] + vy[ip, jp, kp] * dyuiso[i, j, k] + vz[ip, jp, kp] * dzuiso[i, j, k])
