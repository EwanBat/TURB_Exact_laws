from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRvbetadrho(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho", "pm'", "vx", "vy", "vz", 
                      "dxrho'", "dyrho'", "dzrho'")
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'","rho"))
        pmP = sp.symbols(("pm'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        dxrhoP, dyrhoP, dzrhoP = sp.symbols(("dxrho'", "dyrho'", "dzrho'"))
        
        self.expr = rhoNP/rhoP*pmP*(vxNP*dxrhoP+vyNP*dyrhoP+vzNP*dzrhoP)
        
    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pm, dxrho, dyrho, dzrho, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pm, dxrho, dyrho, dzrho)

    def calc_fourier(self, rho, vx, vy, vz, pm, dxrho, dyrho, dzrho, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pm, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pm"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceRvbetadrho()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvbetadrho().expr


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, pm, 
                             dxrho, dyrho, dzrho, f=njit(SourceRvbetadrho().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pmNP, pmP = pm[i, j, k], pm[ip, jp, kp]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    dxrhoP, dyrhoP, dzrhoP = dxrho[ip, jp, kp], dyrho[ip, jp, kp], dzrho[ip, jp, kp]
    dxrhoNP, dyrhoNP, dzrhoNP = dxrho[i, j, k], dyrho[i, j, k], dzrho[i, j, k]

    return (f(rhoP, rhoNP, pmP, vxNP, vyNP, vzNP, dxrhoP, dyrhoP, dzrhoP) 
            + f(rhoNP, rhoP, pmNP, vxP, vyP, vzP, dxrhoNP, dyrhoNP, dzrhoNP) )
    
def calc_with_fourier(rho, vx, vy, vz, pm, dxrho, dyrho, dzrho):
    #A/B'*C'*D*E' + A'/B*C*D'*E 
    frvx = ft.fft(rho*vx)
    frvy = ft.fft(rho*vy)
    frvz = ft.fft(rho*vz)
    fprdx = ft.fft(pm/rho*dxrho)
    fprdy = ft.fft(pm/rho*dyrho)
    fprdz = ft.fft(pm/rho*dzrho)
    
    output = ft.ifft(frvx*np.conj(fprdx)+frvy*np.conj(fprdy)+frvz*np.conj(fprdz)
                     +np.conj(frvx)*fprdx+np.conj(frvy)*fprdy+np.conj(frvz)*fprdz)
    return output/np.size(output)
    