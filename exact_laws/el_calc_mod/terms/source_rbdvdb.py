from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRbdvdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho",
                      "vx'", "vy'", "vz'",
                      "vx", "vy", "vz",
                      "bx", "by", "bz",
                      "divb'"
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
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divbP = sp.symbols(("divb'"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        self.expr = rhoNP * (bxNP * dvx + byNP * dvy + bzNP * dvz) * divbP

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz, divb)
    
    def calc_fourier(self, rho, vx, vy, vz, bx, by, bz, divb, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, bx, by, bz, divb)

    def variables(self) -> List[str]:
        return ["rho", "v", "b", "divb"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr
    
def load():
    return SourceRbdvdb()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRbdvdb().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, bx, by, bz, divb,f=njit(SourceRbdvdb().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divbP, divbNP = divb[ip, jp, kp], divb[i, j, k]

    return (f(rhoNP, vxP, vyP, vzP, vxNP, vyNP, vzNP, bxNP, byNP, bzNP, divbP) 
            + f(rhoP, vxNP, vyNP, vzNP, vxP, vyP, vzP, bxP, byP, bzP, divbNP))

def calc_with_fourier(rho, vx, vy, vz, bx, by, bz, divb):
    #A*dB*C'-A'*dB*C = A*(B'-B)*C'-A'*(B'-B)*C = A*B'*C' + A'B*C - A'*B'*C - A*B*C'
    frbx = ft.fft(rho*bx)
    frby = ft.fft(rho*by)
    frbz = ft.fft(rho*bz)
    fvdx = ft.fft(vx*divb)
    fvdy = ft.fft(vy*divb)
    fvdz = ft.fft(vz*divb)
    fd = ft.fft(divb)
    frbv = ft.fft(rho*bx*vx+rho*by*vy+rho*bz*vz)
    
    output = ft.ifft(fvdx*np.conj(frbx)+fvdy*np.conj(frby)+fvdz*np.conj(frbz)
                     +np.conj(fvdx)*frbx+np.conj(fvdy)*frby+np.conj(fvdz)*frbz
                     -frbv*np.conj(fd)-np.conj(frbv)*fd)
    return output/np.size(output)
    