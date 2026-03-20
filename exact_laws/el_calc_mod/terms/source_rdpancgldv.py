from typing import List
import sympy as sp
from .abstract_term import calc_source_with_numba
from .source_rdpandv import SourceRdpandv, calc_in_point_with_sympy, calc_with_fourier


class SourceRdpancgldv(SourceRdpandv):
    def __init__(self):
        SourceRdpandv.__init__(self)

    def calc(self, vector: List[int], cube_size: List[int],
        rho, pperpcgl, pparcgl, pm, bx, by, bz,
        dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      rho, pperpcgl, pparcgl, pm, bx, by, bz,
                                      dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz)

    def calc_fourier(self, rho, pperpcgl, pparcgl, pm, bx, by, bz,
                     dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, **kwarg) -> List:
        return calc_with_fourier(rho, pperpcgl, pparcgl, pm, bx, by, bz,
                                      dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz)

    def variables(self) -> List[str]:
        return ["rho", "pcgl", "pm", "gradv", "b"]


def load():
    return SourceRdpancgldv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRdpancgldv().expr


