from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class SourcePmvdr2(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx", "vy", "vz", 
                      "rho'", "rho", "pm",
                      "dxrho'", "dyrho'", "dzrho'",
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        pmNP = sp.symbols(("pm"))
        dxrhoP, dyrhoP, dzrhoP = sp.symbols(("dxrho'", "dyrho'", "dzrho'"))
        
        self.expr = rhoNP / rhoP * pmNP * (vxNP * dxrhoP + vyNP * dyrhoP + vzNP * dzrhoP)
    
    def calc(self, vector:List[int], cube_size:List[int],vx, vy, vz, rho, pm, dxrho, dyrho, dzrho, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, rho, pm, dxrho, dyrho, dzrho)

    def calc_fourier(self, vx, vy, vz, rho, pm, dxrho, dyrho, dzrho, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, rho, pm, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ['v', 'rho', 'pm', 'gradrho']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourcePmvdr2()

def print_expr():
    return SourcePmvdr2().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, rho, pm, dxrho, dyrho, dzrho,
                             f=njit(SourcePmvdr2().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pmP, pmNP = pm[ip, jp, kp], pm[i, j, k]
    dxrhoP, dyrhoP, dzrhoP = dxrho[ip, jp, kp], dyrho[ip, jp, kp], dzrho[ip, jp, kp]
    dxrhoNP, dyrhoNP, dzrhoNP = dxrho[i, j, k], dyrho[i, j, k], dzrho[i, j, k]
    
    out = (f(vxNP, vyNP, vzNP, rhoP, rhoNP, pmNP, dxrhoP, dyrhoP, dzrhoP)
           + f(vxP, vyP, vzP, rhoNP, rhoP, pmP, dxrhoNP, dyrhoNP, dzrhoNP))
    
    return out

def calc_with_fourier(vx, vy, vz, rho, pm, dxrho, dyrho, dzrho):
    #A*B*C*D'/E'+A'*B'*C'*D/E
    frpvx = ft.fft(rho*pm*vx)
    frpvy = ft.fft(rho*pm*vy)
    frpvz = ft.fft(rho*pm*vz) 
    fdrx = ft.fft(dxrho/rho)
    fdry = ft.fft(dyrho/rho)
    fdrz = ft.fft(dzrho/rho)
    output = ft.ifft(np.conj(frpvx)*fdrx+np.conj(frpvy)*fdry+np.conj(frpvz)*fdrz
                   + frpvx*np.conj(fdrx)+ frpvy*np.conj(fdry)+ frpvz*np.conj(fdrz))
    return output/np.size(output)
    