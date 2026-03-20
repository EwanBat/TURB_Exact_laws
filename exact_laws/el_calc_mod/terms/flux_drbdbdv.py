from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrbdbdv(AbstractTerm):
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

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
    
        dbx = bxP - bxNP
        dby = byP - byNP
        dbz = bzP - bzNP
        
        drbx = rhoP * bxP - rhoNP * bxNP
        drby = rhoP * byP - rhoNP * byNP
        drbz = rhoP * bzP - rhoNP * bzNP
    
        self.exprx = (drbx * dbx + drby * dby + drbz * dbz) * dvx
        self.expry = (drbx * dbx + drby * dby + drbz * dbz) * dvy
        self.exprz = (drbx * dbx + drby * dby + drbz * dbz) * dvz
    
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
    return FluxDrbdbdv()

def print_expr():
    return FluxDrbdbdv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             vx, vy, vz, 
                             bx, by, bz,  
                             fx=njit(FluxDrbdbdv().fctx),
                             fy=njit(FluxDrbdbdv().fcty),
                             fz=njit(FluxDrbdbdv().fctz)):
    
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
    frbx = ft.fft(rho*bx)
    frby = ft.fft(rho*by)
    frbz = ft.fft(rho*bz)
    frbxbx = ft.fft(rho*bx*bx)
    frbyby = ft.fft(rho*by*by)
    frbzbz = ft.fft(rho*bz*bz)
    
    fvx = ft.fft(vx)
    fvxbx = ft.fft(vx*bx)
    fvxby = ft.fft(vx*by)
    fvxbz = ft.fft(vx*bz)
    frvxbx = ft.fft(rho*vx*bx)
    frvxby = ft.fft(rho*vx*by)
    frvxbz = ft.fft(rho*vx*bz)
    flux_x = ft.ifft(fvx*np.conj(frbxbx+frbyby+frbzbz) - np.conj(fvx)*(frbxbx+frbyby+frbzbz) 
                        + (frbx*np.conj(fvxbx)+frby*np.conj(fvxby)+frbz*np.conj(fvxbz))
                        - (np.conj(frbx)*fvxbx+np.conj(frby)*fvxby+np.conj(frbz)*fvxbz)
                        + (fbx*np.conj(frvxbx)+fby*np.conj(frvxby)+fbz*np.conj(frvxbz))
                        - (np.conj(fbx)*frvxbx+np.conj(fby)*frvxby+np.conj(fbz)*frvxbz))
    del(fvxbx,fvxby,fvxbz,fvx,frvxbx,frvxby,frvxbz)
    
    fvy = ft.fft(vy)
    fbxvy = ft.fft(bx*vy)
    fvyby = ft.fft(vy*by)
    fvybz = ft.fft(vy*bz)
    frbxvy = ft.fft(rho*vy*bx)
    frvyby = ft.fft(rho*vy*by)
    frvybz = ft.fft(rho*vy*bz)
    flux_y = ft.ifft(fvy*np.conj(frbxbx+frbyby+frbzbz) - np.conj(fvy)*(frbxbx+frbyby+frbzbz) 
                        + (frbx*np.conj(fbxvy)+frby*np.conj(fvyby)+frbz*np.conj(fvybz))
                        - (np.conj(frbx)*fbxvy+np.conj(frby)*fvyby+np.conj(frbz)*fvybz)
                        + (fbx*np.conj(frbxvy)+fby*np.conj(frvyby)+fbz*np.conj(frvybz))
                        - (np.conj(fbx)*frbxvy+np.conj(fby)*frvyby+np.conj(fbz)*frvybz))
    del(fbxvy,fvyby,fvybz,fvy,frbxvy,frvyby,frvybz)
    
    fvz = ft.fft(vz)
    fbxvz = ft.fft(bx*vz)
    fbyvz = ft.fft(by*vz)
    fvzbz = ft.fft(vz*bz)
    frbxvz = ft.fft(rho*bx*vz)
    frbyvz = ft.fft(rho*by*vz)
    frvzbz = ft.fft(rho*vz*bz)
    flux_z = ft.ifft(fvz*np.conj(frbxbx+frbyby+frbzbz) - np.conj(fvz)*(frbxbx+frbyby+frbzbz) 
                        + (frbx*np.conj(fbxvz)+frby*np.conj(fbyvz)+frbz*np.conj(fvzbz))
                        - (np.conj(frbx)*fbxvz+np.conj(frby)*fbyvz+np.conj(frbz)*fvzbz)
                        + (fbx*np.conj(frbxvz)+fby*np.conj(frbyvz)+fbz*np.conj(frvzbz))
                        - (np.conj(fbx)*frbxvz+np.conj(fby)*frbyvz+np.conj(fbz)*frvzbz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 