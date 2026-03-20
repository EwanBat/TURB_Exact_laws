from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba
from .source_rduisodv import SourceRduisodv, calc_in_point_with_sympy, calc_with_fourier


class SourceRducgldv(SourceRduisodv):
    def __init__(self):
        SourceRduisodv.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int], rho, ucgl, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, ucgl, divv)

    def calc_fourier(self, rho, ucgl, divv, **kwarg) -> List:
        return calc_with_fourier(rho, ucgl, divv)

    def variables(self) -> List[str]:
        return ["rho", "ucgl", "divv"]


def load():
    return SourceRducgldv()


def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRducgldv().expr
