from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDbdbdj(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("jx'", "jy'", "jz'", "jx", "jy", "jz",
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
        IjxP, IjyP, IjzP = sp.symbols(("jx'", "jy'", "jz'"))
        IjxNP, IjyNP, IjzNP = sp.symbols(("jx", "jy", "jz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))

        dIjx = IjxP - IjxNP
        dIjy = IjyP - IjyNP
        dIjz = IjzP - IjzNP
    
        dIbx = IbxP - IbxNP
        dIby = IbyP - IbyNP
        dIbz = IbzP - IbzNP
    
        self.exprx = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dIjx
        self.expry = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dIjy
        self.exprz = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dIjz
        
    def calc(self, vector:List[int], cube_size:List[int], Ijx, Ijy, Ijz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, Ijx, Ijy, Ijz, Ibx, Iby, Ibz)

    def calc_fourier(self, Ijx, Ijy, Ijz, Ibx, Iby, Ibz, **kwarg) -> List:
        return calc_with_fourier(Ijx, Ijy, Ijz, Ibx, Iby, Ibz)
    
    def variables(self) -> List[str]:
        return ['Ib','Ij']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDbdbdj()

def print_expr():
    return FluxDbdbdj().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                            Ijx, Ijy, Ijz,
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDbdbdj().fctx),
                             fy=njit(FluxDbdbdj().fcty),
                             fz=njit(FluxDbdbdj().fctz)):
    
    IjxP, IjyP, IjzP = Ijx[ip, jp, kp], Ijy[ip, jp, kp], Ijz[ip, jp, kp]
    IjxNP, IjyNP, IjzNP = Ijx[i, j, k], Ijy[i, j, k], Ijz[i, j, k]
    
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outy = fy(
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outz = fz(
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    return outx, outy, outz

def calc_with_fourier(Ijx, Ijy, Ijz, Ibx, Iby, Ibz):
    fbx = ft.fft(Ibx)
    fby = ft.fft(Iby)
    fbz = ft.fft(Ibz)
    fbxbx = ft.fft(Ibx*Ibx)
    fbyby = ft.fft(Iby*Iby)
    fbzbz = ft.fft(Ibz*Ibz)
    
    fjx = ft.fft(Ijx)
    fjxbx = ft.fft(Ijx*Ibx)
    fjxby = ft.fft(Ijx*Iby)
    fjxbz = ft.fft(Ijx*Ibz)
    flux_x = ft.ifft(fjx*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fjx)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fjxbx)+fby*np.conj(fjxby)+fbz*np.conj(fjxbz))
                        - 2*(np.conj(fbx)*fjxbx+np.conj(fby)*fjxby+np.conj(fbz)*fjxbz))
    del(fjxbx,fjxby,fjxbz,fjx)
    
    fjy = ft.fft(Ijy)
    fbxjy = ft.fft(Ibx*Ijy)
    fjyby = ft.fft(Ijy*Iby)
    fjybz = ft.fft(Ijy*Ibz)
    flux_y = ft.ifft(fjy*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fjy)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fbxjy)+fby*np.conj(fjyby)+fbz*np.conj(fjybz))
                        - 2*(np.conj(fbx)*fbxjy+np.conj(fby)*fjyby+np.conj(fbz)*fjybz))
    del(fbxjy,fjyby,fjybz,fjy)
    
    fjz = ft.fft(Ijz)
    fbxjz = ft.fft(Ibx*Ijz)
    fbyjz = ft.fft(Iby*Ijz)
    fjzbz = ft.fft(Ijz*Ibz)
    flux_z = ft.ifft(fjz*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fjz)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fbxjz)+fby*np.conj(fbyjz)+fbz*np.conj(fjzbz))
                        - 2*(np.conj(fbx)*fbxjz+np.conj(fby)*fbyjz+np.conj(fbz)*fjzbz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 
    