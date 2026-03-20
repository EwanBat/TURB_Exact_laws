from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrduisodv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho", "uiso'", "uiso",    
            "vx'", "vy'", "vz'", "vx", "vy", "vz")
        
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
        rhoP, rhoNP = sp.symbols(("rho'","rho"))
        uisoP, uisoNP = sp.symbols(("uiso'","uiso"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        
        dr = rhoP - rhoNP
        duiso = uisoP - uisoNP
        
        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        self.exprx = dr * duiso * dvx
        self.expry = dr * duiso * dvy
        self.exprz = dr * duiso * dvz
    
    def calc(self, vector:List[int], cube_size:List[int], rho, uiso, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, uiso, vx, vy, vz)

    def calc_fourier(self, rho, uiso, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, uiso, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','uiso', 'v']

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz
    
def load():
    return FluxDrduisodv()

def print_expr():
    return FluxDrduisodv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, uiso, vx, vy, vz,
                             fx=njit(FluxDrduisodv().fctx),
                             fy=njit(FluxDrduisodv().fcty),
                             fz=njit(FluxDrduisodv().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    uisoP, uisoNP = uiso[ip, jp, kp], uiso[i, j, k]
        
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP, uisoP, uisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outy = fy(
        rhoP, rhoNP, uisoP, uisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outz = fz(
        rhoP, rhoNP, uisoP, uisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz
    
def calc_with_fourier(rho, uiso, vx, vy, vz):
    fr = ft.fft(rho) 
    fu = ft.fft(uiso) 
    fru = ft.fft(rho*uiso) 
    
    fvx = ft.fft(vx) 
    frvx = ft.fft(rho*vx)
    fuvx = ft.fft(uiso*vx)
    flux_x = ft.ifft(np.conj(fru)*fvx - fru*np.conj(fvx) 
                     + np.conj(fuvx)*fr - fuvx*np.conj(fr)
                     + np.conj(frvx)*fu - frvx*np.conj(fu))
    del(fvx,frvx,fuvx)
    
    fvy = ft.fft(vy) 
    frvy = ft.fft(rho*vy)
    fuvy = ft.fft(uiso*vy)
    flux_y = ft.ifft(np.conj(fru)*fvy - fru*np.conj(fvy) 
                     + np.conj(fuvy)*fr - fuvy*np.conj(fr)
                     + np.conj(frvy)*fu - frvy*np.conj(fu))
    del(fvy,frvy,fuvy)
    
    fvz = ft.fft(vz) 
    frvz = ft.fft(rho*vz)
    fuvz = ft.fft(uiso*vz)
    flux_z = ft.ifft(np.conj(fru)*fvz - fru*np.conj(fvz) 
                     + np.conj(fuvz)*fr - fuvz*np.conj(fr)
                     + np.conj(frvz)*fu - frvz*np.conj(fu))

    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 