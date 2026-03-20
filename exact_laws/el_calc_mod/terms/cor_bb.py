from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class DissBinc(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("bx'", "by'", "bz'", "bx", "by", "bz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("bx'", "by'", "bz'"))
        vxNP, vyNP, vzNP = sp.symbols(("bx", "by", "bz"))

        self.expr =  (vxP-vxNP)*(vxP-vxNP) + (vyP-vyNP)*(vyP-vyNP) + (vzP-vzNP)*(vzP-vzNP) 

    def calc(self, vector: List[int], cube_size: List[int],  Ibx, Iby, Ibz,  **kwarg
        ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, Ibx, Iby, Ibz, )
    
    def calc_fourier(self, Ibx, Iby, Ibz, **kwarg) -> List:
        return calc_with_fourier(Ibx, Iby, Ibz, )

    def variables(self) -> List[str]:
        return ["Ib",]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return DissBinc()


def print_expr():
    return DissBinc().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             bx, by, bz, 
                             f=njit(DissBinc().fct)):
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    return f(bxP, byP, bzP, bxNP, byNP, bzNP)

def calc_with_fourier(bx, by, bz):
    fbx = ft.fft(bx)
    fby = ft.fft(by)
    fbz = ft.fft(bz)
    
    output = ft.ifft(fbx*np.conj(fbx) + fby*np.conj(fby) + fbz*np.conj(fbz))
    return output/np.size(output)
    
