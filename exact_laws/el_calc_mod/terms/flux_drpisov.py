from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj

class FluxDrpisov(AbstractTerm):
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
        
        rpNP = (rhoP + rhoNP) * pisoNP
        rpP = (rhoP + rhoNP) * pisoP
        
        self.exprx = rpNP * vxP - rpP * vxNP
        self.expry = rpNP * vyP - rpP * vyNP
        self.exprz = rpNP * vzP - rpP * vzNP
    
    def calc(self, vector:List[int], cube_size:List[int], rho, piso, vx, vy, vz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, piso, vx, vy, vz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, piso, vx, vy, vz)
    
    def calc_fourier(self, rho, piso, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, piso, vx, vy, vz, traj=traj)

    def variables(self) -> List[str]:
        return ['rho','piso', 'v']

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz
    
def load():
    return FluxDrpisov()

def print_expr():
    return FluxDrpisov().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, piso, vx, vy, vz,
                             fx=njit(FluxDrpisov().fctx),
                             fy=njit(FluxDrpisov().fcty),
                             fz=njit(FluxDrpisov().fctz)):
    
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

@njit
def calc_in_point_with_sympy_traj(t, tp, rho, piso, vx, vy, vz,
                                 fx=njit(FluxDrpisov().fctx),
                                 fy=njit(FluxDrpisov().fcty),
                                 fz=njit(FluxDrpisov().fctz)):
    rhoP, rhoNP = rho[tp], rho[t]

    pisoP, pisoNP = piso[tp], piso[t]

    vxP, vyP, vzP = vx[tp], vy[tp], vz[tp]
    vxNP, vyNP, vzNP = vx[t], vy[t], vz[t]

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
    
def calc_with_fourier(rho, piso, vx, vy, vz, traj=False):
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    fp = transform(piso)
    frp = transform(rho*piso)
    
    fvx = transform(vx)
    frvx = transform(rho*vx)
    flux_x = inv_transform(np.conj(frp)*fvx - frp*np.conj(fvx)
                     - np.conj(frvx)*fp + frvx*np.conj(fp))
    del(fvx,frvx)
    
    fvy = transform(vy)
    frvy = transform(rho*vy)
    flux_y = inv_transform(np.conj(frp)*fvy - frp*np.conj(fvy) 
                     - np.conj(frvy)*fp + frvy*np.conj(fp))
    del(fvy,frvy)
    
    fvz = transform(vz)
    frvz = transform(rho*vz)
    flux_z = inv_transform(np.conj(frp)*fvz - frp*np.conj(fvz) 
                     - np.conj(frvz)*fp + frvz*np.conj(fp))
    
    if traj:
        return [flux_x/np.size(flux_x,axis=-1),flux_y/np.size(flux_y,axis=-1),flux_z/np.size(flux_z,axis=-1)]
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 
        