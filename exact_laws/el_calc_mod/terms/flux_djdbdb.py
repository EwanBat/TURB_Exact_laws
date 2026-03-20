from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba

class FluxDjdbdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("jx'", "jy'", "jz'",
                 "jx", "jy", "jz",
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
    
        self.exprx = (dIjx * dIbx + dIjy * dIby + dIjz * dIbz) * dIbx
        self.expry = (dIjx * dIbx + dIjy * dIby + dIjz * dIbz) * dIby
        self.exprz = (dIjx * dIbx + dIjy * dIby + dIjz * dIbz) * dIbz
        
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
    return FluxDjdbdb()

def print_expr():
    return FluxDjdbdb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             Ijx, Ijy, Ijz, 
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDjdbdb().fctx),
                             fy=njit(FluxDjdbdb().fcty),
                             fz=njit(FluxDjdbdb().fctz)):
    
    IjxP, IjyP, IjzP = Ijx[ip, jp, kp], Ijy[ip, jp, kp], Ijz[ip, jp, kp]
    IjxNP, IjyNP, IjzNP = Ijx[i, j, k], Ijy[i, j, k], Ijz[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outy = fy(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outz = fz(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    
    return outx, outy, outz

def calc_with_fourier(Ijx, Ijy, Ijz, Ibx, Iby, Ibz):
    fIjx = ft.fft(Ijx)
    fIjy = ft.fft(Ijy)
    fIjz = ft.fft(Ijz)
    fbx = ft.fft(Ibx)
    fby = ft.fft(Iby)
    fbz = ft.fft(Ibz)
    fbxbz = ft.fft(Ibx*Ibz)
    fIjxbx = ft.fft(Ijx*Ibx)
    fIjyby = ft.fft(Ijy*Iby)
    fIjzbz = ft.fft(Ijz*Ibz)
    
    fbxby = ft.fft(Ibx*Iby)
    fbxIjy = ft.fft(Ibx*Ijy)
    fbxIjz = ft.fft(Ibx*Ijz)
    fbxbx = ft.fft(Ibx*Ibx)
    flux_x = ft.ifft(fbx*np.conj(fIjxbx+fIjyby+fIjzbz) - np.conj(fbx)*(fIjxbx+fIjyby+fIjzbz) 
                        + (fbx*np.conj(fIjxbx)+fby*np.conj(fbxIjy)+fbz*np.conj(fbxIjz))
                        - (np.conj(fbx)*fIjxbx+np.conj(fby)*fbxIjy+np.conj(fbz)*fbxIjz)
                        + (fIjx*np.conj(fbxbx)+fIjy*np.conj(fbxby)+fIjz*np.conj(fbxbz))
                        - (np.conj(fIjx)*fbxbx+np.conj(fIjy)*fbxby+np.conj(fIjz)*fbxbz))
    del(fbxIjy,fbxIjz,fbxbx)
    
    fbybz = ft.fft(Iby*Ibz)
    fbyby = ft.fft(Iby*Iby)
    fIjxby = ft.fft(Ijx*Iby)
    fbyIjz = ft.fft(Iby*Ijz)
    flux_y = ft.ifft(fby*np.conj(fIjxbx+fIjyby+fIjzbz) - np.conj(fby)*(fIjxbx+fIjyby+fIjzbz) 
                        + (fbx*np.conj(fIjxby)+fby*np.conj(fIjyby)+fbz*np.conj(fbyIjz))
                        - (np.conj(fbx)*fIjxby+np.conj(fby)*fIjyby+np.conj(fbz)*fbyIjz)
                        + (fIjx*np.conj(fbxby)+fIjy*np.conj(fbyby)+fIjz*np.conj(fbybz))
                        - (np.conj(fIjx)*fbxby+np.conj(fIjy)*fbyby+np.conj(fIjz)*fbybz))
    del(fbyby,fIjxby,fbyIjz,fbxby)
    
    fIjxbz = ft.fft(Ijx*Ibz)
    fIjybz = ft.fft(Ijy*Ibz)
    fbzbz = ft.fft(Ibz*Ibz)
    flux_z = ft.ifft(fbz*np.conj(fIjxbx+fIjyby+fIjzbz) - np.conj(fbz)*(fIjxbx+fIjyby+fIjzbz) 
                        + (fbx*np.conj(fIjxbz)+fby*np.conj(fIjybz)+fbz*np.conj(fIjzbz))
                        - (np.conj(fbx)*fIjxbz+np.conj(fby)*fIjybz+np.conj(fbz)*fIjzbz)
                        + (fIjx*np.conj(fbxbz)+fIjy*np.conj(fbybz)+fIjz*np.conj(fbzbz))
                        - (np.conj(fIjx)*fbxbz+np.conj(fIjy)*fbybz+np.conj(fIjz)*fbzbz))
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 