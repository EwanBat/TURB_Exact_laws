from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

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
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)

    def calc_fourier(self, rho, vx, vy, vz, bx, by, bz, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, bx, by, bz)

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

def calc_with_fourier(rho, vx, vy, vz, bx, by, bz):    
    fbx = ft.fft(bx)
    fby = ft.fft(by)
    fbz = ft.fft(bz)
    frvx = ft.fft(rho*vx)
    frvy = ft.fft(rho*vy)
    frvz = ft.fft(rho*vz)
    frvxbx = ft.fft(rho*vx*bx)
    frvyby = ft.fft(rho*vy*by)
    frvzbz = ft.fft(rho*vz*bz)
    
    fbxbx = ft.fft(bx*bx)
    fbybx = ft.fft(by*bx)
    fbzbx = ft.fft(bz*bx)
    frvybx = ft.fft(rho*vy*bx)
    frvzbx = ft.fft(rho*vz*bx)
    flux_x = ft.ifft(fbx*np.conj(frvxbx+frvyby+frvzbz) - np.conj(fbx)*(frvxbx+frvyby+frvzbz) 
                        + (frvx*np.conj(fbxbx)+frvy*np.conj(fbybx)+frvz*np.conj(fbzbx))
                        - (np.conj(frvx)*fbxbx+np.conj(frvy)*fbybx+np.conj(frvz)*fbzbx)
                        + (fbx*np.conj(frvxbx)+fby*np.conj(frvybx)+fbz*np.conj(frvzbx))
                        - (np.conj(fbx)*frvxbx+np.conj(fby)*frvybx+np.conj(fbz)*frvzbx))
    del(frvybx,frvzbx,fbxbx)
    
    fbyby = ft.fft(by*by)
    fbzby = ft.fft(bz*by)
    frvxby = ft.fft(rho*vx*by)
    frvzby = ft.fft(rho*vz*by)
    flux_y = ft.ifft(fby*np.conj(frvxbx+frvyby+frvzbz) - np.conj(fby)*(frvxbx+frvyby+frvzbz) 
                        + (frvx*np.conj(fbybx)+frvy*np.conj(fbyby)+frvz*np.conj(fbzby))
                        - (np.conj(frvx)*fbybx+np.conj(frvy)*fbyby+np.conj(frvz)*fbzby)
                        + (fbx*np.conj(frvxby)+fby*np.conj(frvyby)+fbz*np.conj(frvzby))
                        - (np.conj(fbx)*frvxby+np.conj(fby)*frvyby+np.conj(fbz)*frvzby))
    del(frvxby,frvzby,fbyby,fbybx)
    
    fbzbz = ft.fft(bz*bz)
    frvxbz = ft.fft(rho*vx*bz)
    frvybz = ft.fft(rho*vy*bz)
    flux_z = ft.ifft(fbz*np.conj(frvxbx+frvyby+frvzbz) - np.conj(fbz)*(frvxbx+frvyby+frvzbz) 
                        + (frvx*np.conj(fbzbx)+frvy*np.conj(fbzby)+frvz*np.conj(fbzbz))
                        - (np.conj(frvx)*fbzbx+np.conj(frvy)*fbzby+np.conj(frvz)*fbzbz)
                        + (fbx*np.conj(frvxbz)+fby*np.conj(frvybz)+fbz*np.conj(frvzbz))
                        - (np.conj(fbx)*frvxbz+np.conj(fby)*frvybz+np.conj(fbz)*frvzbz))
    
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 