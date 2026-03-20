from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDbdbdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "vx", "vy", "vz",
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
    
        self.exprx = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvx
        self.expry = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvy
        self.exprz = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvz
        
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)

    def calc_fourier(self, vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz)
    
    def variables(self) -> List[str]:
        return ['Ib','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDbdbdv()

def print_expr():
    return FluxDbdbdv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDbdbdv().fctx),
                             fy=njit(FluxDbdbdv().fcty),
                             fz=njit(FluxDbdbdv().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outy = fy(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outz = fz(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    return outx, outy, outz

def calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz):    
    fbx = ft.fft(Ibx)
    fby = ft.fft(Iby)
    fbz = ft.fft(Ibz)
    fbxbx = ft.fft(Ibx*Ibx)
    fbyby = ft.fft(Iby*Iby)
    fbzbz = ft.fft(Ibz*Ibz)
    
    fvx = ft.fft(vx)
    fvxbx = ft.fft(vx*Ibx)
    fvxby = ft.fft(vx*Iby)
    fvxbz = ft.fft(vx*Ibz)
    flux_x = ft.ifft(fvx*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fvx)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fvxbx)+fby*np.conj(fvxby)+fbz*np.conj(fvxbz))
                        - 2*(np.conj(fbx)*fvxbx+np.conj(fby)*fvxby+np.conj(fbz)*fvxbz))
    del(fvxbx,fvxby,fvxbz,fvx)
    
    fvy = ft.fft(vy)
    fbxvy = ft.fft(Ibx*vy)
    fvyby = ft.fft(vy*Iby)
    fvybz = ft.fft(vy*Ibz)
    flux_y = ft.ifft(fvy*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fvy)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fbxvy)+fby*np.conj(fvyby)+fbz*np.conj(fvybz))
                        - 2*(np.conj(fbx)*fbxvy+np.conj(fby)*fvyby+np.conj(fbz)*fvybz))
    del(fbxvy,fvyby,fvybz,fvy)
    
    fvz = ft.fft(vz)
    fbxvz = ft.fft(Ibx*vz)
    fbyvz = ft.fft(Iby*vz)
    fvzbz = ft.fft(vz*Ibz)
    flux_z = ft.ifft(fvz*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fvz)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fbxvz)+fby*np.conj(fbyvz)+fbz*np.conj(fvzbz))
                        - 2*(np.conj(fbx)*fbxvz+np.conj(fby)*fbyvz+np.conj(fbz)*fvzbz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 