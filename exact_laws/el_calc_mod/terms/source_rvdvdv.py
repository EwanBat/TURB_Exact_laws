from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRvdvdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho",
                      "vx'", "vy'", "vz'",
                      "vx", "vy", "vz",
                      "divv'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
      
    def set_sympy_expr(self):  
        rhoNP = sp.symbols(("rho"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        divvP = sp.symbols(("divv'"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        self.expr = rhoNP * (vxNP * dvx + vyNP * dvy + vzNP * dvz) * divvP

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, divv)
    
    def calc_fourier(self, rho, vx, vy, vz, divv, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, divv)

    def variables(self) -> List[str]:
        return ["rho", "v", "divv"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceRvdvdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvdvdv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, divv,f=njit(SourceRvdvdv().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]

    return (f(rhoNP, vxP, vyP, vzP, vxNP, vyNP, vzNP, divvP) 
            + f(rhoP, vxNP, vyNP, vzNP, vxP, vyP, vzP, divvNP))

def calc_with_fourier(rho, vx, vy, vz, divv):
    #A*dB*C'-A'*dB*C = A*(B'-B)*C'-A'*(B'-B)*C = A*B'*C' + A'B*C - A'*B'*C - A*B*C'
    frvx = ft.fft(rho*vx)
    frvy = ft.fft(rho*vy)
    frvz = ft.fft(rho*vz)
    fvdx = ft.fft(vx*divv)
    fvdy = ft.fft(vy*divv)
    fvdz = ft.fft(vz*divv)
    fd = ft.fft(divv)
    frvv = ft.fft(rho*vx*vx+rho*vy*vy+rho*vz*vz)
    
    output = ft.ifft(fvdx*np.conj(frvx)+fvdy*np.conj(frvy)+fvdz*np.conj(frvz)
                     +np.conj(fvdx)*frvx+np.conj(fvdy)*frvy+np.conj(fvdz)*frvz
                     -frvv*np.conj(fd)-np.conj(frvv)*fd)
    return output/np.size(output)
