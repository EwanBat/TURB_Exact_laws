from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj

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

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso)

    def calc_fourier(self, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, traj=traj)

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

@njit
def calc_in_point_with_sympy_traj(t, tp, rho, vx, vy, vz, pm, piso, dxuiso, dyuiso, dzuiso, f=njit(SourceRvbetadu().fct)):
    rhoP, rhoNP = rho[tp], rho[t]
    pmNP, pmP = pm[t], pm[tp]
    pisoNP, pisoP = piso[t], piso[tp]
    vxP, vyP, vzP = vx[tp], vy[tp], vz[tp]
    vxNP, vyNP, vzNP = vx[t], vy[t], vz[t]
    dxuisoP, dyuisoP, dzuisoP = dxuiso[tp], dyuiso[tp], dzuiso[tp]
    dxuisoNP, dyuisoNP, dzuisoNP = dxuiso[t], dyuiso[t], dzuiso[t]

    return (f(rhoNP, pmP, pisoP, vxNP, vyNP, vzNP, dxuisoP, dyuisoP, dzuisoP)
        + f(rhoP, pmNP, pisoNP, vxP, vyP, vzP, dxuisoNP, dyuisoNP, dzuisoNP) )
    
def calc_with_fourier(rho, vx, vy, vz, pm, piso, 
                             dxuiso, dyuiso, dzuiso, traj=False):
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    #AB'/C'*D*E' + A'B/C*D'*E 
    frvx = transform(rho*vx)
    frvy = transform(rho*vy)
    frvz = transform(rho*vz)
    fpdx = transform(pm/piso*dxuiso)
    fpdy = transform(pm/piso*dyuiso)
    fpdz = transform(pm/piso*dzuiso)
    
    return inv_transform(frvx*np.conj(fpdx)+frvy*np.conj(fpdy)+frvz*np.conj(fpdz)
                     +np.conj(frvx)*fpdx+np.conj(frvy)*fpdy+np.conj(frvz)*fpdz)
    