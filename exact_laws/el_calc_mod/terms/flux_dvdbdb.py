from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDvdbdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'",
                 "vx", "vy", "vz",
                 "bx'", "by'", "bz'",
                 "bx", "by", "bz",
                )
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
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
    
        dIbx = IbxP - IbxNP
        dIby = IbyP - IbyNP
        dIbz = IbzP - IbzNP
    
        self.exprx = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbx
        self.expry = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIby
        self.exprz = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbz
        
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        #return calc_flux_with_numba(calc_in_point, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)
    
    def calc_fourier(self, vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz)

    def variables(self) -> List[str]:
        return ['Ib','v']

def load():
    return FluxDvdbdb()

def print_expr():
    sp.init_printing(use_latex=True)
    return FluxDvdbdb().exprx, FluxDvdbdb().expry, FluxDvdbdb().exprz

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDvdbdb().fctx),
                             fy=njit(FluxDvdbdb().fcty),
                             fz=njit(FluxDvdbdb().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outy = fy(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outz = fz(
        vxP, vyP, vzP, 
        vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    
    return outx, outy, outz

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz, Ibx, Iby, Ibz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    dIbx = Ibx[ip,jp,kp] - Ibx[i,j,k]
    dIby = Iby[ip,jp,kp] - Iby[i,j,k]
    dIbz = Ibz[ip,jp,kp] - Ibz[i,j,k]
    
    fx = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbx
    fy = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIby
    fz = (dvx * dIbx + dvy * dIby + dvz * dIbz) * dIbz
    
    return fx, fy, fz

def calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz):
        fvx = ft.fft(vx)
        fvy = ft.fft(vy)
        fvz = ft.fft(vz)
        fbx = ft.fft(Ibx)
        fby = ft.fft(Iby)
        fbz = ft.fft(Ibz)
        fbxbz = ft.fft(Ibx*Ibz)
        fvxbx = ft.fft(vx*Ibx)
        fvyby = ft.fft(vy*Iby)
        fvzbz = ft.fft(vz*Ibz)
        
        fbxby = ft.fft(Ibx*Iby)
        fbxvy = ft.fft(Ibx*vy)
        fbxvz = ft.fft(Ibx*vz)
        fbxbx = ft.fft(Ibx*Ibx)
        flux_x = ft.ifft(fbx*np.conj(fvxbx+fvyby+fvzbz) - np.conj(fbx)*(fvxbx+fvyby+fvzbz) 
                         + (fbx*np.conj(fvxbx)+fby*np.conj(fbxvy)+fbz*np.conj(fbxvz))
                         - (np.conj(fbx)*fvxbx+np.conj(fby)*fbxvy+np.conj(fbz)*fbxvz)
                         + (fvx*np.conj(fbxbx)+fvy*np.conj(fbxby)+fvz*np.conj(fbxbz))
                         - (np.conj(fvx)*fbxbx+np.conj(fvy)*fbxby+np.conj(fvz)*fbxbz))
        del(fbxvy,fbxvz,fbxbx)
        
        fbybz = ft.fft(Iby*Ibz)
        fbyby = ft.fft(Iby*Iby)
        fvxby = ft.fft(vx*Iby)
        fbyvz = ft.fft(Iby*vz)
        flux_y = ft.ifft(fby*np.conj(fvxbx+fvyby+fvzbz) - np.conj(fby)*(fvxbx+fvyby+fvzbz) 
                         + (fbx*np.conj(fvxby)+fby*np.conj(fvyby)+fbz*np.conj(fbyvz))
                         - (np.conj(fbx)*fvxby+np.conj(fby)*fvyby+np.conj(fbz)*fbyvz)
                         + (fvx*np.conj(fbxby)+fvy*np.conj(fbyby)+fvz*np.conj(fbybz))
                         - (np.conj(fvx)*fbxby+np.conj(fvy)*fbyby+np.conj(fvz)*fbybz))
        del(fbyby,fvxby,fbyvz,fbxby)
        
        fvxbz = ft.fft(vx*Ibz)
        fvybz = ft.fft(vy*Ibz)
        fbzbz = ft.fft(Ibz*Ibz)
        flux_z = ft.ifft(fbz*np.conj(fvxbx+fvyby+fvzbz) - np.conj(fbz)*(fvxbx+fvyby+fvzbz) 
                         + (fbx*np.conj(fvxbz)+fby*np.conj(fvybz)+fbz*np.conj(fvzbz))
                         - (np.conj(fbx)*fvxbz+np.conj(fby)*fvybz+np.conj(fbz)*fvzbz)
                         + (fvx*np.conj(fbxbz)+fvy*np.conj(fbybz)+fvz*np.conj(fbzbz))
                         - (np.conj(fvx)*fbxbz+np.conj(fvy)*fbybz+np.conj(fvz)*fbzbz))
        return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 