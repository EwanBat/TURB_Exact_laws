from typing import List
from numba import njit
import numpy as np
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceTest(AbstractTerm):
    def __init__(self):
        pass

    def calc(self, vector: List[int], cube_size: List[int], divv, uiso, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point, *vector, *cube_size,divv, uiso)

    def variables(self) -> List[str]:
        return ['divv', 'uiso']


def load():
    return SourceTest()


@njit
def calc_in_point(i, j, k, ip, jp, kp, divv, uiso):
    return (divv[i, j, k] * uiso[ip, jp, kp]
            + divv[ip, jp, kp] * uiso[i, j, k] )
