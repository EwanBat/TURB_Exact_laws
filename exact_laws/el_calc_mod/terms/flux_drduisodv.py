from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj

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
    
    def calc(self, vector:List[int], cube_size:List[int], rho, uiso, vx, vy, vz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, uiso, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, uiso, vx, vy, vz)

    def calc_fourier(self, rho, uiso, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, uiso, vx, vy, vz, traj=traj)

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

@njit
def calc_in_point_with_sympy_traj(t, tp, rho, uiso, vx, vy, vz,
                                 fx=njit(FluxDrduisodv().fctx),
                                 fy=njit(FluxDrduisodv().fcty),
                                 fz=njit(FluxDrduisodv().fctz)):
    rhoP, rhoNP = rho[tp], rho[t]

    uisoP, uisoNP = uiso[tp], uiso[t]

    vxP, vyP, vzP = vx[tp], vy[tp], vz[tp]
    vxNP, vyNP, vzNP = vx[t], vy[t], vz[t]

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
    
def calc_with_fourier(rho, uiso, vx, vy, vz, traj=False):
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    fr = transform(rho)
    fu = transform(uiso)
    fru = transform(rho*uiso)
    
    fvx = transform(vx)
    frvx = transform(rho*vx)
    fuvx = transform(uiso*vx)
    flux_x = inv_transform(np.conj(fru)*fvx - fru*np.conj(fvx) 
                     + np.conj(fuvx)*fr - fuvx*np.conj(fr)
                     + np.conj(frvx)*fu - frvx*np.conj(fu))
    del(fvx,frvx,fuvx)
    
    fvy = transform(vy)
    frvy = transform(rho*vy)
    fuvy = transform(uiso*vy)
    flux_y = inv_transform(np.conj(fru)*fvy - fru*np.conj(fvy) 
                     + np.conj(fuvy)*fr - fuvy*np.conj(fr)
                     + np.conj(frvy)*fu - frvy*np.conj(fu))
    del(fvy,frvy,fuvy)
    
    fvz = transform(vz)
    frvz = transform(rho*vz)
    fuvz = transform(uiso*vz)
    flux_z = inv_transform(np.conj(fru)*fvz - fru*np.conj(fvz) 
                     + np.conj(fuvz)*fr - fuvz*np.conj(fr)
                     + np.conj(frvz)*fu - frvz*np.conj(fu))

    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 