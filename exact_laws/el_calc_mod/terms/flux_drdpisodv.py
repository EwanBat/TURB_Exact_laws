from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpisodv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho", "piso'", "piso",    
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
        pisoP, pisoNP = sp.symbols(("piso'","piso"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        
        dr = rhoP - rhoNP
        dpiso = pisoP - pisoNP
        
        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        self.exprx = dr * dpiso * dvx
        self.expry = dr * dpiso * dvy
        self.exprz = dr * dpiso * dvz
    
    def calc(self, vector:List[int], cube_size:List[int], rho, piso, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, piso, vx, vy, vz)

    def calc_fourier(self, rho, piso, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, piso, vx, vy, vz)
    
    def variables(self) -> List[str]:
        return ['rho','piso', 'v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrdpisodv()

def print_expr():
    return FluxDrdpisodv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, piso, vx, vy, vz,
                             fx=njit(FluxDrdpisodv().fctx),
                             fy=njit(FluxDrdpisodv().fcty),
                             fz=njit(FluxDrdpisodv().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    pisoP, pisoNP = piso[ip, jp, kp], piso[i, j, k]
        
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP, pisoP, pisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outy = fy(
        rhoP, rhoNP, pisoP, pisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outz = fz(
        rhoP, rhoNP, pisoP, pisoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz

def calc_with_fourier(rho, piso, vx, vy, vz):
    fr = ft.fft(rho) 
    fp = ft.fft(piso) 
    frp = ft.fft(rho*piso) 
    
    fvx = ft.fft(vx) 
    frvx = ft.fft(rho*vx)
    fpvx = ft.fft(piso*vx)
    flux_x = ft.ifft(np.conj(frp)*fvx - frp*np.conj(fvx) 
                     + np.conj(fpvx)*fr - fpvx*np.conj(fr)
                     + np.conj(frvx)*fp - frvx*np.conj(fp))
    del(fvx,frvx,fpvx)
    
    fvy = ft.fft(vy) 
    frvy = ft.fft(rho*vy)
    fpvy = ft.fft(piso*vy)
    flux_y = ft.ifft(np.conj(frp)*fvy - frp*np.conj(fvy) 
                     + np.conj(fpvy)*fr - fpvy*np.conj(fr)
                     + np.conj(frvy)*fp - frvy*np.conj(fp))
    del(fvy,frvy,fpvy)
    
    fvz = ft.fft(vz) 
    frvz = ft.fft(rho*vz)
    fpvz = ft.fft(piso*vz)
    flux_z = ft.ifft(np.conj(frp)*fvz - frp*np.conj(fvz) 
                     + np.conj(fpvz)*fr - fpvz*np.conj(fr)
                     + np.conj(frvz)*fp - frvz*np.conj(fp))

    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 