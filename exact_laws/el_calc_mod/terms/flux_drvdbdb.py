from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj

class FluxDrvdbdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
            "vx'", "vy'", "vz'", "vx", "vy", "vz",
            "bx'", "by'", "bz'", "bx", "by", "bz")
        
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
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
    
        dbx = bxP - bxNP
        dby = byP - byNP
        dbz = bzP - bzNP
        
        drvx = rhoP * vxP - rhoNP * vxNP
        drvy = rhoP * vyP - rhoNP * vyNP
        drvz = rhoP * vzP - rhoNP * vzNP
    
        self.exprx = (drvx * dbx + drvy * dby + drvz * dbz) * dbx
        self.expry = (drvx * dbx + drvy * dby + drvz * dbz) * dby
        self.exprz = (drvx * dbx + drvy * dby + drvz * dbz) * dbz
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, bx, by, bz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)

    def calc_fourier(self, rho, vx, vy, vz, bx, by, bz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, bx, by, bz, traj=traj)

    def variables(self) -> List[str]:
        return ['rho','b','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrvdbdb()

def print_expr():
    return FluxDrvdbdb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             vx, vy, vz, 
                             bx, by, bz,  
                             fx=njit(FluxDrvdbdb().fctx),
                             fy=njit(FluxDrvdbdb().fcty),
                             fz=njit(FluxDrvdbdb().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    outy = fy(
        rhoP,rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    outz = fz(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    return outx, outy, outz

@njit
def calc_in_point_with_sympy_traj(t, tp,
                             rho,
                             vx, vy, vz,
                             bx, by, bz,
                             fx=njit(FluxDrvdbdb().fctx),
                             fy=njit(FluxDrvdbdb().fcty),
                             fz=njit(FluxDrvdbdb().fctz)):
    rhoP, rhoNP = rho[tp], rho[t]

    vxP, vyP, vzP = vx[tp], vy[tp], vz[tp]
    vxNP, vyNP, vzNP = vx[t], vy[t], vz[t]

    bxP, byP, bzP = bx[tp], by[tp], bz[tp]
    bxNP, byNP, bzNP = bx[t], by[t], bz[t]

    outx = fx(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        bxP, byP, bzP, bxNP, byNP, bzNP)

    outy = fy(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        bxP, byP, bzP, bxNP, byNP, bzNP)

    outz = fz(
        rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        bxP, byP, bzP, bxNP, byNP, bzNP)

    return outx, outy, outz

def calc_with_fourier(rho, vx, vy, vz, bx, by, bz, traj=False):    
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    fbx = transform(bx)
    fby = transform(by)
    fbz = transform(bz)
    frvx = transform(rho*vx)
    frvy = transform(rho*vy)
    frvz = transform(rho*vz)
    frvxbx = transform(rho*vx*bx)
    frvyby = transform(rho*vy*by)
    frvzbz = transform(rho*vz*bz)
    
    fbxbx = transform(bx*bx)
    fbybx = transform(by*bx)
    fbzbx = transform(bz*bx)
    frvybx = transform(rho*vy*bx)
    frvzbx = transform(rho*vz*bx)
    flux_x = inv_transform(fbx*np.conj(frvxbx+frvyby+frvzbz) - np.conj(fbx)*(frvxbx+frvyby+frvzbz) 
                        + (frvx*np.conj(fbxbx)+frvy*np.conj(fbybx)+frvz*np.conj(fbzbx))
                        - (np.conj(frvx)*fbxbx+np.conj(frvy)*fbybx+np.conj(frvz)*fbzbx)
                        + (fbx*np.conj(frvxbx)+fby*np.conj(frvybx)+fbz*np.conj(frvzbx))
                        - (np.conj(fbx)*frvxbx+np.conj(fby)*frvybx+np.conj(fbz)*frvzbx))
    del(frvybx,frvzbx,fbxbx)
    
    fbyby = transform(by*by)
    fbzby = transform(bz*by)
    frvxby = transform(rho*vx*by)
    frvzby = transform(rho*vz*by)
    flux_y = inv_transform(fby*np.conj(frvxbx+frvyby+frvzbz) - np.conj(fby)*(frvxbx+frvyby+frvzbz) 
                        + (frvx*np.conj(fbybx)+frvy*np.conj(fbyby)+frvz*np.conj(fbzby))
                        - (np.conj(frvx)*fbybx+np.conj(frvy)*fbyby+np.conj(frvz)*fbzby)
                        + (fbx*np.conj(frvxby)+fby*np.conj(frvyby)+fbz*np.conj(frvzby))
                        - (np.conj(fbx)*frvxby+np.conj(fby)*frvyby+np.conj(fbz)*frvzby))
    del(frvxby,frvzby,fbyby,fbybx)
    
    fbzbz = transform(bz*bz)
    frvxbz = transform(rho*vx*bz)
    frvybz = transform(rho*vy*bz)
    flux_z = inv_transform(fbz*np.conj(frvxbx+frvyby+frvzbz) - np.conj(fbz)*(frvxbx+frvyby+frvzbz) 
                        + (frvx*np.conj(fbzbx)+frvy*np.conj(fbzby)+frvz*np.conj(fbzbz))
                        - (np.conj(frvx)*fbzbx+np.conj(frvy)*fbzby+np.conj(frvz)*fbzbz)
                        + (fbx*np.conj(frvxbz)+fby*np.conj(frvybz)+fbz*np.conj(frvzbz))
                        - (np.conj(fbx)*frvxbz+np.conj(fby)*frvybz+np.conj(fbz)*frvzbz))
    
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 