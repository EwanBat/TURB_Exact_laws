from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class SourceRvbetadu(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho", "pm'", "piso'", "vx", "vy", "vz", 
                      "dxuiso'", "dyuiso'", "dzuiso'")
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        rhoNP = sp.symbols(("rho"))
        pmP, pisoP = sp.symbols(("pm'","piso'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        dxuisoP, dyuisoP, dzuisoP = sp.symbols(("dxuiso'", "dyuiso'", "dzuiso'"))
        
        self.expr = rhoNP*pmP/pisoP*(vxNP*dxuisoP+vyNP*dyuisoP+vzNP*dzuisoP)

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso)

    def calc_fourier(self, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso)

    def variables(self) -> List[str]:
        return ["rho", "graduiso", "v", "pm", "piso"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceRvbetadu()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvbetadu().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, pm, piso, 
                             dxuiso, dyuiso, dzuiso, f=njit(SourceRvbetadu().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pmNP, pmP = pm[i, j, k], pm[ip, jp, kp]
    pisoNP, pisoP = piso[i, j, k], piso[ip, jp, kp]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    dxuisoP, dyuisoP, dzuisoP = dxuiso[ip, jp, kp], dyuiso[ip, jp, kp], dzuiso[ip, jp, kp]
    dxuisoNP, dyuisoNP, dzuisoNP = dxuiso[i, j, k], dyuiso[i, j, k], dzuiso[i, j, k]

    return (f(rhoNP, pmP, pisoP, vxNP, vyNP, vzNP, dxuisoP, dyuisoP, dzuisoP) 
            + f(rhoP, pmNP, pisoNP, vxP, vyP, vzP, dxuisoNP, dyuisoNP, dzuisoNP) )
    
def calc_with_fourier(rho, vx, vy, vz, pm, piso, 
                             dxuiso, dyuiso, dzuiso):
    #AB'/C'*D*E' + A'B/C*D'*E 
    frvx = ft.fft(rho*vx)
    frvy = ft.fft(rho*vy)
    frvz = ft.fft(rho*vz)
    fpdx = ft.fft(pm/piso*dxuiso)
    fpdy = ft.fft(pm/piso*dyuiso)
    fpdz = ft.fft(pm/piso*dzuiso)
    
    return ft.ifft(frvx*np.conj(fpdx)+frvy*np.conj(fpdy)+frvz*np.conj(fpdz)
                     +np.conj(frvx)*fpdx+np.conj(frvy)*fpdy+np.conj(frvz)*fpdz)
    