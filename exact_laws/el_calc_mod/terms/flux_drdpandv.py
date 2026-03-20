from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrdpandv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ( "rho'", "rho",
            "ppar'", "ppar", "pperp'", "pperp", "pm'", "pm",    
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
        pparP, pparNP = sp.symbols(("ppar'","ppar"))
        pperpP, pperpNP = sp.symbols(("pperp'","pperp"))
        pmP, pmNP = sp.symbols(("pm'","pm"))
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))

        dr = rhoP - rhoNP
        pressP = (pparP - pperpP) / (2*pmP)
        pressNP = (pparNP - pperpNP) / (2*pmNP)
        
        dpxx = pressP * bxP * bxP - pressNP * bxNP * bxNP
        dpyy = pressP * byP * byP - pressNP * byNP * byNP
        dpzz = pressP * bzP * bzP - pressNP * bzNP * bzNP
        dpxy = pressP * bxP * byP - pressNP * bxNP * byNP
        dpxz = pressP * bxP * bzP - pressNP * bxNP * bzNP
        dpyz = pressP * byP * bzP - pressNP * byNP * bzNP    
        
        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
        
        self.exprx = dr * (dpxx * dvx + dpxy * dvy + dpxz * dvz)
        self.expry = dr * (dpxy * dvx + dpyy * dvy + dpyz * dvz)
        self.exprz = dr * (dpxz * dvx + dpyz * dvy + dpzz * dvz)
    
    def calc(self, vector:List[int], cube_size:List[int], rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz)

    def calc_fourier(self, rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz, **kwarg) -> List:
        return calc_with_fourier(rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz)
    
    def variables(self) -> List[str]:
        return ['rho','pgyr', 'pm', 'v', 'b']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrdpandv()

def print_expr():
    return FluxDrdpandv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho, pperp, ppar, 
                             vx, vy, vz, 
                             pm, bx, by, bz,                       
                             fx=njit(FluxDrdpandv().fctx),
                             fy=njit(FluxDrdpandv().fcty),
                             fz=njit(FluxDrdpandv().fctz)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    pparP, pparNP = ppar[ip, jp, kp], ppar[i, j, k]
    pperpP, pperpNP = pperp[ip, jp, kp], pperp[i, j, k]
    pmP, pmNP = pm[ip, jp, kp], pm[i, j, k]
        
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    outx = fx(
        rhoP, rhoNP,
        pparP, pparNP, pperpP, pperpNP, pmP, pmNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    outy = fy(
        rhoP,rhoNP,
        pparP, pparNP, pperpP, pperpNP, pmP, pmNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    outz = fz(
        rhoP, rhoNP,
        pparP, pparNP, pperpP, pperpNP, pmP, pmNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        bxP, byP, bzP, bxNP, byNP, bzNP)
    
    return outx, outy, outz

def calc_with_fourier(rho, pperp, ppar, vx, vy, vz, pm, bx, by, bz):
    fr = ft.fft(rho) 
    fvx = ft.fft(vx) 
    fvy = ft.fft(vy) 
    fvz = ft.fft(vz) 
    frvx = ft.fft(rho*vx) 
    frvy = ft.fft(rho*vy) 
    frvz = ft.fft(rho*vz) 
    
    pxx = (ppar - pperp) / (2*pm) * bx * bx
    pxy = (ppar - pperp) / (2*pm) * bx * by
    pxz = (ppar - pperp) / (2*pm) * bx * bz
    fpvx = ft.fft(pxx*vx+pxy*vy+pxz*vz) 
    fpxx = ft.fft(pxx)
    fpxy = ft.fft(pxy)
    fpxz = ft.fft(pxz)
    frpxx = ft.fft(rho*pxx)
    frpxy = ft.fft(rho*pxy)
    frpxz = ft.fft(rho*pxz)
    flux_x = ft.ifft(- np.conj(fr)*fpvx + fr*np.conj(fpvx)
                     - (frpxx*np.conj(fvx) + frpxy*np.conj(fvy) + frpxz*np.conj(fvz))
                     + (np.conj(frpxx)*fvx + np.conj(frpxy)*fvy + np.conj(frpxz)*fvz)
                     - (frvx*np.conj(fpxx) + frvy*np.conj(fpxy) + frvz*np.conj(fpxz))
                     + (np.conj(frvx)*fpxx + np.conj(frvy)*fpxy + np.conj(frvz)*fpxz))
    del(fpvx,fpxx,frpxx,pxx)
    
    pyy = (ppar - pperp) / (2*pm) * by * by
    pyz = (ppar - pperp) / (2*pm) * by * bz
    fpvy = ft.fft(pxy*vx+pyy*vy+pyz*vz) 
    fpyy = ft.fft(pyy)
    fpyz = ft.fft(pyz)
    frpyy = ft.fft(rho*pyy)
    frpyz = ft.fft(rho*pyz)
    flux_y = ft.ifft(- np.conj(fr)*fpvy + fr*np.conj(fpvy)
                     - (frpxy*np.conj(fvx) + frpyy*np.conj(fvy) + frpyz*np.conj(fvz))
                     + (np.conj(frpxy)*fvx + np.conj(frpyy)*fvy + np.conj(frpyz)*fvz)
                     - (frvx*np.conj(fpxy) + frvy*np.conj(fpyy) + frvz*np.conj(fpyz))
                     + (np.conj(frvx)*fpxy + np.conj(frvy)*fpyy + np.conj(frvz)*fpyz))
    del(fpvy,fpyy,frpyy,pyy,pxy,fpxy,frpxy)
    
    pzz = (ppar - pperp) / (2*pm) * bz * bz
    fpvz = ft.fft(pxz*vx+pyz*vy+pzz*vz) 
    fpzz = ft.fft(pzz)
    frpzz = ft.fft(rho*pzz)
    flux_z = ft.ifft(- np.conj(fr)*fpvz + fr*np.conj(fpvz)
                     - (frpxz*np.conj(fvx) + frpyz*np.conj(fvy) + frpzz*np.conj(fvz))
                     + (np.conj(frpxz)*fvx + np.conj(frpyz)*fvy + np.conj(frpzz)*fvz)
                     - (frvx*np.conj(fpxz) + frvy*np.conj(fpyz) + frvz*np.conj(fpzz))
                     + (np.conj(frvx)*fpxz + np.conj(frvy)*fpyz + np.conj(frvz)*fpzz))
    
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 
    
        