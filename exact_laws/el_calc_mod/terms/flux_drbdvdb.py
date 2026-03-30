from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDrbdvdb(AbstractTerm):
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
    
        self.exprx = (drbx * dvx + drby * dvy + drbz * dvz) * dbx
        self.expry = (drbx * dvx + drby * dvy + drbz * dvz) * dby
        self.exprz = (drbx * dvx + drby * dvy + drbz * dvz) * dbz
    
    def calc(self, vector:List[int], cube_size:List[int], rho, vx, vy, vz, bx, by, bz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, bx, by, bz)

    def calc_fourier(self, rho, vx, vy, vz, bx, by, bz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, bx, by, bz, traj=traj)
    
    def variables(self) -> List[str]:
        return ['rho','b','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDrbdvdb()

def print_expr():
    return FluxDrbdvdb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp,
                             rho, 
                             vx, vy, vz, 
                             bx, by, bz,  
                             fx=njit(FluxDrbdvdb().fctx),
                             fy=njit(FluxDrbdvdb().fcty),
                             fz=njit(FluxDrbdvdb().fctz)):
    
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
    
def calc_with_fourier(rho, vx, vy, vz, bx, by, bz, traj=False):    
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    fvx = transform(vx)
    fvy = transform(vy)
    fvz = transform(vz)
    frbx = transform(rho*bx)
    frby = transform(rho*by)
    frbz = transform(rho*bz)
    frbxvx = transform(rho*bx*vx)
    frbyvy = transform(rho*by*vy)
    frbzvz = transform(rho*bz*vz)

    fbx = transform(bx)
    fvxbx = transform(vx*bx)
    fvybx = transform(vy*bx)
    fvzbx = transform(vz*bx)
    frbxbx = transform(rho*bx*bx)
    frbybx = transform(rho*by*bx)
    frbzbx = transform(rho*bz*bx)
    flux_x = inv_transform(fbx*np.conj(frbxvx+frbyvy+frbzvz) - np.conj(fbx)*(frbxvx+frbyvy+frbzvz) 
                        + (frbx*np.conj(fvxbx)+frby*np.conj(fvybx)+frbz*np.conj(fvzbx))
                        - (np.conj(frbx)*fvxbx+np.conj(frby)*fvybx+np.conj(frbz)*fvzbx)
                        + (fvx*np.conj(frbxbx)+fvy*np.conj(frbybx)+fvz*np.conj(frbzbx))
                        - (np.conj(fvx)*frbxbx+np.conj(fvy)*frbybx+np.conj(fvz)*frbzbx))
    del(fvxbx,fvybx,fvzbx,fbx,frbxbx)
    
    fby = transform(by)
    fvxby = transform(vx*by)
    fvyby = transform(vy*by)
    fvzby = transform(vz*by)
    frbyby = transform(rho*by*by)
    frbzby = transform(rho*bz*by)
    flux_y = inv_transform(fby*np.conj(frbxvx+frbyvy+frbzvz) - np.conj(fby)*(frbxvx+frbyvy+frbzvz) 
                        + (frbx*np.conj(fvxby)+frby*np.conj(fvyby)+frbz*np.conj(fvzby))
                        - (np.conj(frbx)*fvxby+np.conj(frby)*fvyby+np.conj(frbz)*fvzby)
                        + (fvx*np.conj(frbybx)+fvy*np.conj(frbyby)+fvz*np.conj(frbzby))
                        - (np.conj(fvx)*frbybx+np.conj(fvy)*frbyby+np.conj(fvz)*frbzby))
    del(fvxby,fvyby,fvzby,fby,frbyby,frbybx)
    
    fbz = transform(bz)
    fvxbz = transform(vx*bz)
    fvybz = transform(vy*bz)
    fvzbz = transform(vz*bz)
    frbzbz = transform(rho*bz*bz)
    flux_z = inv_transform(fbz*np.conj(frbxvx+frbyvy+frbzvz) - np.conj(fbz)*(frbxvx+frbyvy+frbzvz) 
                        + (frbx*np.conj(fvxbz)+frby*np.conj(fvybz)+frbz*np.conj(fvzbz))
                        - (np.conj(frbx)*fvxbz+np.conj(frby)*fvybz+np.conj(frbz)*fvzbz)
                        + (fvx*np.conj(frbzbx)+fvy*np.conj(frbzby)+fvz*np.conj(frbzbz))
                        - (np.conj(fvx)*frbzbx+np.conj(fvy)*frbzby+np.conj(fvz)*frbzbz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 