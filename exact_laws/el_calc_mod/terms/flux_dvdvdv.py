from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDvdvdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'",
                 "vx", "vy", "vz"
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

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
    
        self.exprx = (dvx * dvx + dvy * dvy + dvz * dvz) * dvx
        self.expry = (dvx * dvx + dvy * dvy + dvz * dvz) * dvy
        self.exprz = (dvx * dvx + dvy * dvy + dvz * dvz) * dvz
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz)
    
    def calc_fourier(self, vx, vy, vz, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz)

    def variables(self) -> List[str]:
        return ['v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDvdvdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return FluxDvdvdv().exprx, FluxDvdvdv().expry, FluxDvdvdv().exprz

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz,  
                             fx=njit(FluxDvdvdv().fctx),
                             fy=njit(FluxDvdvdv().fcty),
                             fz=njit(FluxDvdvdv().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    outy = fy(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    outz = fz(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = (dvx * dvx + dvy * dvy + dvz * dvz) * dvx
    fy = (dvx * dvx + dvy * dvy + dvz * dvz) * dvy
    fz = (dvx * dvx + dvy * dvy + dvz * dvz) * dvz
    
    return fx, fy, fz

def calc_with_fourier(vx, vy, vz):
    fvx = ft.fft(vx)
    fvy = ft.fft(vy)
    fvz = ft.fft(vz)
    fvxvx = ft.fft(vx*vx)
    fvyvy = ft.fft(vy*vy)
    fvzvz = ft.fft(vz*vz)
    fvxvy = ft.fft(vx*vy)
    fvxvz = ft.fft(vx*vz)
    flux_x = ft.ifft(fvx*np.conj(fvxvx+fvyvy+fvzvz) - np.conj(fvx)*(fvxvx+fvyvy+fvzvz) 
                        + 2*(fvx*np.conj(fvxvx)+fvy*np.conj(fvxvy)+fvz*np.conj(fvxvz))
                        - 2*(np.conj(fvx)*fvxvx+np.conj(fvy)*fvxvy+np.conj(fvz)*fvxvz))
    fvyvz = ft.fft(vy*vz)
    flux_y = ft.ifft(fvy*np.conj(fvxvx+fvyvy+fvzvz) - np.conj(fvy)*(fvxvx+fvyvy+fvzvz) 
                        + 2*(fvx*np.conj(fvxvy)+fvy*np.conj(fvyvy)+fvz*np.conj(fvyvz))
                        - 2*(np.conj(fvx)*fvxvy+np.conj(fvy)*fvyvy+np.conj(fvz)*fvyvz))
    del(fvxvy)
    flux_z = ft.ifft(fvz*np.conj(fvxvx+fvyvy+fvzvz) - np.conj(fvz)*(fvxvx+fvyvy+fvzvz) 
                        + 2*(fvx*np.conj(fvxvz)+fvy*np.conj(fvyvz)+fvz*np.conj(fvzvz))
                        - 2*(np.conj(fvx)*fvxvz+np.conj(fvy)*fvyvz+np.conj(fvz)*fvzvz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 
