from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrvdvdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
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
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
    
        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        drvx = rhoP * vxP - rhoNP * vxNP
        drvy = rhoP * vyP - rhoNP * vyNP
        drvz = rhoP * vzP - rhoNP * vzNP
    
        self.exprx = (drvx * dvx + drvy * dvy + drvz * dvz) * dvx
        self.expry = (drvx * dvx + drvy * dvy + drvz * dvz) * dvy
        self.exprz = (drvx * dvx + drvy * dvy + drvz * dvz) * dvz
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz)

    def calc_fourier(self, rho, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz)

    def variables(self) -> List[str]:
        return ['rho','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrvdvdv()

def print_expr():
    return FluxDrvdvdv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             vx, vy, vz,   
                             fx=njit(FluxDrvdvdv().fctx),
                             fy=njit(FluxDrvdvdv().fcty),
                             fz=njit(FluxDrvdvdv().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outy = fy(
        rhoP,rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    outz = fz(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz

def calc_with_fourier(rho, vx, vy, vz):
    fvx = ft.fft(vx)
    fvy = ft.fft(vy)
    fvz = ft.fft(vz)
    frvx = ft.fft(rho*vx)
    frvy = ft.fft(rho*vy)
    frvz = ft.fft(rho*vz)
    frvxvx = ft.fft(rho*vx*vx)
    frvyvy = ft.fft(rho*vy*vy)
    frvzvz = ft.fft(rho*vz*vz)
    
    fvxvx = ft.fft(vx*vx)
    fvyvx = ft.fft(vy*vx)
    fvzvx = ft.fft(vz*vx)
    frvyvx = ft.fft(rho*vy*vx)
    frvzvx = ft.fft(rho*vz*vx)
    flux_x = ft.ifft(fvx*np.conj(frvxvx+frvyvy+frvzvz) - np.conj(fvx)*(frvxvx+frvyvy+frvzvz) 
                        + (frvx*np.conj(fvxvx)+frvy*np.conj(fvyvx)+frvz*np.conj(fvzvx))
                        - (np.conj(frvx)*fvxvx+np.conj(frvy)*fvyvx+np.conj(frvz)*fvzvx)
                        + (fvx*np.conj(frvxvx)+fvy*np.conj(frvyvx)+fvz*np.conj(frvzvx))
                        - (np.conj(fvx)*frvxvx+np.conj(fvy)*frvyvx+np.conj(fvz)*frvzvx))
    del(fvxvx)
    
    fvyvy = ft.fft(vy*vy)
    fvzvy = ft.fft(vz*vy)
    frvzvy = ft.fft(rho*vz*vy)
    flux_y = ft.ifft(fvy*np.conj(frvxvx+frvyvy+frvzvz) - np.conj(fvy)*(frvxvx+frvyvy+frvzvz) 
                        + (frvx*np.conj(fvyvx)+frvy*np.conj(fvyvy)+frvz*np.conj(fvzvy))
                        - (np.conj(frvx)*fvyvx+np.conj(frvy)*fvyvy+np.conj(frvz)*fvzvy)
                        + (fvx*np.conj(frvyvx)+fvy*np.conj(frvyvy)+fvz*np.conj(frvzvy))
                        - (np.conj(fvx)*frvyvx+np.conj(fvy)*frvyvy+np.conj(fvz)*frvzvy))
    del(fvyvy,fvyvx,frvyvx)
    
    fvzvz = ft.fft(vz*vz)
    flux_z = ft.ifft(fvz*np.conj(frvxvx+frvyvy+frvzvz) - np.conj(fvz)*(frvxvx+frvyvy+frvzvz) 
                        + (frvx*np.conj(fvzvx)+frvy*np.conj(fvzvy)+frvz*np.conj(fvzvz))
                        - (np.conj(frvx)*fvzvx+np.conj(frvy)*fvzvy+np.conj(frvz)*fvzvz)
                        + (fvx*np.conj(frvzvx)+fvy*np.conj(frvzvy)+fvz*np.conj(frvzvz))
                        - (np.conj(fvx)*frvzvx+np.conj(fvy)*frvzvy+np.conj(fvz)*frvzvz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)]   