from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrpmv2(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "rho'", "pm'",
                 "vx", "vy", "vz", "rho", "pm"
                )
        self.fctx = sp.lambdify(
            sp.symbols(quantities),
            self.exprx,
            "numpy",
        )
        self.fcty = sp.lambdify(
            sp.symbols(quantities),
            self.expry,
            "numpy",
        )
        self.fctz = sp.lambdify(
            sp.symbols(quantities),
            self.exprz,
            "numpy",
        )
        
    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        pmP, pmNP = sp.symbols(("pm'", "pm"))
        
        self.exprx = rhoNP * pmP * vxP - rhoP * pmNP * vxNP
        self.expry = rhoNP * pmP * vyP - rhoP * pmNP * vyNP
        self.exprz = rhoNP * pmP * vzP - rhoP * pmNP * vzNP
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, rho, pm, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, rho, pm)

    def calc_fourier(self, vx, vy, vz, rho, pm, **kwarg) -> List:
        return calc_with_fourier( vx, vy, vz, rho, pm)

    def variables(self) -> List[str]:
        return ['v', 'rho', 'pm']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrpmv2()

def print_expr():
    return FluxDrpmv2().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz,  rho, pm, 
                             fx=njit(FluxDrpmv2().fctx),
                             fy=njit(FluxDrpmv2().fcty),
                             fz=njit(FluxDrpmv2().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pmP, pmNP = pm[ip, jp, kp], pm[i, j, k]
    
    outx = fx(vxP, vyP, vzP, rhoP, pmP, vxNP, vyNP, vzNP, rhoNP, pmNP)
    outy = fy(vxP, vyP, vzP, rhoP, pmP, vxNP, vyNP, vzNP, rhoNP, pmNP)
    outz = fz(vxP, vyP, vzP, rhoP, pmP, vxNP, vyNP, vzNP, rhoNP, pmNP)
    
    return outx, outy, outz


def calc_with_fourier( vx, vy, vz, rho, pm):
    fr = ft.fft(rho) 
        
    fpvx = ft.fft(pm*vx) 
    flux_x = ft.ifft(np.conj(fr)*fpvx - fr*np.conj(fpvx))
     
    fpvy = ft.fft(pm*vy)
    flux_y = ft.ifft(np.conj(fr)*fpvy - fr*np.conj(fpvy))
    
    fpvz = ft.fft(pm*vz)
    flux_z = ft.ifft(np.conj(fr)*fpvz - fr*np.conj(fpvz))
    
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 