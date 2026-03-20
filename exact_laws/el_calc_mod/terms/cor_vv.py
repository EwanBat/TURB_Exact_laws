from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class DissVinc(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("vx'", "vy'", "vz'", "vx", "vy", "vz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))

        self.expr =  (vxP-vxNP)*(vxP-vxNP) + (vyP-vyNP)*(vyP-vyNP) + (vzP-vzNP)*(vzP-vzNP) 

    def calc(self, vector: List[int], cube_size: List[int],  vx, vy, vz,  **kwarg
        ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, )
    
    def calc_fourier(self, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, )

    def variables(self) -> List[str]:
        return ["v",]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return DissVinc()


def print_expr():
    return DissVinc().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             f=njit(DissVinc().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    return f(vxP, vyP, vzP, vxNP, vyNP, vzNP)

def calc_with_fourier(vx, vy, vz):
    fvx = ft.fft(vx)
    fvy = ft.fft(vy)
    fvz = ft.fft(vz)
    
    output = ft.ifft(fvx*np.conj(fvx) + fvy*np.conj(fvy) + fvz*np.conj(fvz))
    return output/np.size(output)
    