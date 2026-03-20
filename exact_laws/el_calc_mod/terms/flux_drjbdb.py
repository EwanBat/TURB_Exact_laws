from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrjbdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
                      "bx'", "by'", "bz'", "bx", "by", "bz",
                      "jx'", "jy'", "jz'", "jx", "jy", "jz")
        
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
        jxP, jyP, jzP = sp.symbols(("jx'", "jy'", "jz'"))
        jxNP, jyNP, jzNP = sp.symbols(("jx", "jy", "jz"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))

        drjbx = (rhoP * (jyP * bzP - byP * jzP) + rhoNP * (jyNP * bzNP - byNP * jzNP))/2
        drjby = (rhoP * (jzP * bxP - bzP * jxP) + rhoNP * (jzNP * bxNP - bzNP * jxNP))/2
        drjbz = (rhoP * (jxP * byP - bxP * jyP) + rhoNP * (jxNP * byNP - bxNP * jyNP))/2
    
        dbx = bxP - bxNP
        dby = byP - byNP
        dbz = bzP - bzNP
    
        self.exprx = drjby * dbz - drjbz * dby
        self.expry = drjbz * dbx - drjbx * dbz
        self.exprz = drjbx * dby - drjby * dbx
    
    def calc(self, vector:List[int], cube_size:List[int], rho, bx, by, bz, jx, jy, jz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, jx, jy, jz)

    def calc_fourier(self, rho, bx, by, bz, jx, jy, jz, **kwarg) -> List:
        return calc_with_fourier(rho, bx, by, bz, jx, jy, jz)

    def variables(self) -> List[str]:
        return ['rho','b','j']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrjbdb()

def print_expr():
    return FluxDrjbdb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             bx, by, bz, 
                             jx, jy, jz,  
                             fx=njit(FluxDrjbdb().fctx),
                             fy=njit(FluxDrjbdb().fcty),
                             fz=njit(FluxDrjbdb().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    jxP, jyP, jzP = jx[ip, jp, kp], jy[ip, jp, kp], jz[ip, jp, kp]
    jxNP, jyNP, jzNP = jx[i, j, k], jy[i, j, k], jz[i, j, k]
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP, 
        jxP, jyP, jzP, jxNP, jyNP, jzNP)
    
    outy = fy(
        rhoP,rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP, 
        jxP, jyP, jzP, jxNP, jyNP, jzNP)
    
    outz = fz(
        rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP, 
        jxP, jyP, jzP, jxNP, jyNP, jzNP)
    
    return outx, outy, outz
    
def calc_with_fourier(rho, bx, by, bz, jx, jy, jz):    
    rjbx = rho * jy * bz - rho * by * jz
    rjby = rho * jz * bx - rho * bz * jx
    rjbz = rho * jx * by - rho * bx * jy
    
    frjbx = ft.fft(rjbx)
    frjby = ft.fft(rjby)
    frjbz = ft.fft(rjbz) 
    del(rjbx,rjby,rjbz)
    
    fbx = ft.fft(bx)
    fby = ft.fft(by)
    fbz = ft.fft(bz)
    
    flux_x = ft.ifft((np.conj(frjby)*fbz-frjby*np.conj(fbz))-(np.conj(frjbz)*fby-frjbz*np.conj(fby)))/2 
    flux_y = ft.ifft((np.conj(frjbz)*fbx-frjbz*np.conj(fbx))-(np.conj(frjbx)*fbz-frjbx*np.conj(fbz)))/2
    flux_z = ft.ifft((np.conj(frjbx)*fby-frjbx*np.conj(fby))-(np.conj(frjby)*fbx-frjby*np.conj(fbx)))/2
    
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)]  
        