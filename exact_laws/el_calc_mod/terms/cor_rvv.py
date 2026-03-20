
from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class CorRvv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("rho'","rho",
                 "vx'", "vy'", "vz'", "vx", "vy", "vz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        
        psvv = vxP*vxNP + vyP*vyNP + vzP*vzNP

        self.expr = (rhoP+rhoNP)*psvv/2 /2  

    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz)

    def calc_fourier(self, rho, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz)
        

    def variables(self) -> List[str]:
        return ["v", "rho"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return CorRvv()


def print_expr():
    return CorRvv().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho,
                             vx, vy, vz, 
                             f=njit(CorRvv().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    return f(rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP
    )

def calc_with_fourier(rho, vx, vy, vz):
    fvx = ft.fft(vx)
    fvy = ft.fft(vy)
    fvz = ft.fft(vz)
    frhovx = ft.fft(rho*vx)
    frhovy = ft.fft(rho*vy)
    frhovz = ft.fft(rho*vz)
    
    output = ft.ifft(frhovx*np.conj(fvx) + frhovy*np.conj(fvy) + frhovz*np.conj(fvz)
                    + np.conj(frhovx)*fvx + np.conj(frhovy)*fvy + np.conj(frhovz)*fvz)/4
    return output/np.size(output)