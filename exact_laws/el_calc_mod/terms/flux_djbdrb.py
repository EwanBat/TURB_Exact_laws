from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDjbdrb(AbstractTerm):
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

        djbx = (jyP * bzP - byP * jzP) - (jyNP * bzNP - byNP * jzNP)
        djby = (jzP * bxP - bzP * jxP) - (jzNP * bxNP - bzNP * jxNP)
        djbz = (jxP * byP - bxP * jyP) - (jxNP * byNP - bxNP * jyNP)
    
        drbx = (rhoP * bxP + rhoNP * bxNP)/2
        drby = (rhoP * byP + rhoNP * byNP)/2
        drbz = (rhoP * bzP + rhoNP * bzNP)/2
    
        self.exprx = djby * drbz - djbz * drby
        self.expry = djbz * drbx - djbx * drbz
        self.exprz = djbx * drby - djby * drbx
    
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
    return FluxDjbdrb()

def print_expr():
    return FluxDjbdrb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             bx, by, bz, 
                             jx, jy, jz,  
                             fx=njit(FluxDjbdrb().fctx),
                             fy=njit(FluxDjbdrb().fcty),
                             fz=njit(FluxDjbdrb().fctz)):
    
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
    jbx = jy * bz - by * jz
    jby = jz * bx - bz * jx
    jbz = jx * by - bx * jy
    
    fjbx = ft.fft(jbx)
    fjby = ft.fft(jby)
    fjbz = ft.fft(jbz) 
    del(jbx,jby,jbz)
    
    frbx = ft.fft(rho*bx)
    frby = ft.fft(rho*by)
    frbz = ft.fft(rho*bz)
    
    flux_x = ft.ifft((-np.conj(fjby)*frbz+fjby*np.conj(frbz))-(-np.conj(fjbz)*frby+fjbz*np.conj(frby)))/2 
    flux_y = ft.ifft((-np.conj(fjbz)*frbx+fjbz*np.conj(frbx))-(-np.conj(fjbx)*frbz+fjbx*np.conj(frbz)))/2
    flux_z = ft.ifft((-np.conj(fjbx)*frby+fjbx*np.conj(frby))-(-np.conj(fjby)*frbx+fjby*np.conj(frbx)))/2
    
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 
