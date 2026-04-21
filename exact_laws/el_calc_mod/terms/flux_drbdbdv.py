from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj

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
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, bx, by, bz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)
        else:
            return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)

    def calc_fourier(self, rho, vx, vy, vz, bx, by, bz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, bx, by, bz, traj=traj)
    
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

@njit
def calc_in_point_with_sympy_traj(t, tp,
                             rho,
                             vx, vy, vz,
                             bx, by, bz,
                             fx=njit(FluxDrbdbdv().fctx),
                             fy=njit(FluxDrbdbdv().fcty),
                             fz=njit(FluxDrbdbdv().fctz)):
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
    frbx = transform(rho*bx)
    frby = transform(rho*by)
    frbz = transform(rho*bz)
    frbxbx = transform(rho*bx*bx)
    frbyby = transform(rho*by*by)
    frbzbz = transform(rho*bz*bz)

    fvx = transform(vx)
    fvxbx = transform(vx*bx)
    fvxby = transform(vx*by)
    fvxbz = transform(vx*bz)
    frvxbx = transform(rho*vx*bx)
    frvxby = transform(rho*vx*by)
    frvxbz = transform(rho*vx*bz)
    flux_x = inv_transform(fvx*np.conj(frbxbx+frbyby+frbzbz) - np.conj(fvx)*(frbxbx+frbyby+frbzbz) 
                        + (frbx*np.conj(fvxbx)+frby*np.conj(fvxby)+frbz*np.conj(fvxbz))
                        - (np.conj(frbx)*fvxbx+np.conj(frby)*fvxby+np.conj(frbz)*fvxbz)
                        + (fbx*np.conj(frvxbx)+fby*np.conj(frvxby)+fbz*np.conj(frvxbz))
                        - (np.conj(fbx)*frvxbx+np.conj(fby)*frvxby+np.conj(fbz)*frvxbz))
    del(fvxbx,fvxby,fvxbz,fvx,frvxbx,frvxby,frvxbz)
    
    fvy = transform(vy)
    fbxvy = transform(bx*vy)
    fvyby = transform(vy*by)
    fvybz = transform(vy*bz)
    frbxvy = transform(rho*vy*bx)
    frvyby = transform(rho*vy*by)
    frvybz = transform(rho*vy*bz)
    flux_y = inv_transform(fvy*np.conj(frbxbx+frbyby+frbzbz) - np.conj(fvy)*(frbxbx+frbyby+frbzbz) 
                        + (frbx*np.conj(fbxvy)+frby*np.conj(fvyby)+frbz*np.conj(fvybz))
                        - (np.conj(frbx)*fbxvy+np.conj(frby)*fvyby+np.conj(frbz)*fvybz)
                        + (fbx*np.conj(frbxvy)+fby*np.conj(frvyby)+fbz*np.conj(frvybz))
                        - (np.conj(fbx)*frbxvy+np.conj(fby)*frvyby+np.conj(fbz)*frvybz))
    del(fbxvy,fvyby,fvybz,fvy,frbxvy,frvyby,frvybz)
    
    fvz = transform(vz)
    fbxvz = transform(bx*vz)
    fbyvz = transform(by*vz)
    fvzbz = transform(vz*bz)
    frbxvz = transform(rho*bx*vz)
    frbyvz = transform(rho*by*vz)
    frvzbz = transform(rho*vz*bz)
    flux_z = inv_transform(fvz*np.conj(frbxbx+frbyby+frbzbz) - np.conj(fvz)*(frbxbx+frbyby+frbzbz) 
                        + (frbx*np.conj(fbxvz)+frby*np.conj(fbyvz)+frbz*np.conj(fvzbz))
                        - (np.conj(frbx)*fbxvz+np.conj(frby)*fbyvz+np.conj(frbz)*fvzbz)
                        + (fbx*np.conj(frbxvz)+fby*np.conj(frbyvz)+fbz*np.conj(frvzbz))
                        - (np.conj(fbx)*frbxvz+np.conj(fby)*frbyvz+np.conj(fbz)*frvzbz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 