from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class Bg17Vbj(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("vx'", "vy'", "vz'", "vx", "vy", "vz",
                 "bx'", "by'", "bz'", "bx", "by", "bz",
                 "jx'", "jy'", "jz'", "jx", "jy", "jz"
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))
        IjxP, IjyP, IjzP = sp.symbols(("jx'", "jy'", "jz'"))
        IjxNP, IjyNP, IjzNP = sp.symbols(("jx", "jy", "jz"))

        djx = IjxP - IjxNP
        djy = IjyP - IjyNP
        djz = IjzP - IjzNP

        vXbxP = vyP * IbzP - vzP * IbyP
        vXbyP = vzP * IbxP - vxP * IbzP
        vXbzP = vxP * IbyP - vyP * IbxP
        vXbxNP = vyNP * IbzNP - vzNP * IbyNP
        vXbyNP = vzNP * IbxNP - vxNP * IbzNP
        vXbzNP = vxNP * IbyNP - vyNP * IbxNP

        self.expr = (vXbxP - vXbxNP) * djx + (vXbyP - vXbyNP) * djy + (vXbzP - vXbzNP) * djz
    
    def calc(
        self, vector: List[int], cube_size: List[int], vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz)
    
    def calc_fourier(self, vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz)
    
    def variables(self) -> List[str]:
        return ['Ij','Ib','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return Bg17Vbj()

def print_expr():
    return Bg17Vbj().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz, 
                             Ijx, Ijy, Ijz, 
                             f=njit(Bg17Vbj().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    IjxP, IjyP, IjzP = Ijx[ip, jp, kp], Ijy[ip, jp, kp], Ijz[ip, jp, kp]
    IjxNP, IjyNP, IjzNP = Ijx[i, j, k], Ijy[i, j, k], Ijz[i, j, k]
    
    return f(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP, 
        IjxP, IjyP, IjzP, IjxNP, IjyNP, IjzNP
    )
    
def calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz, Ijx, Ijy, Ijz):
    fjx = ft.fft(Ijx)
    fjy = ft.fft(Ijy)
    fjz = ft.fft(Ijz)
    
    vXbx = vy * Ibz - vz * Iby
    vXby = vz * Ibx - vx * Ibz
    vXbz = vx * Iby - vy * Ibx
    
    fvXbx = ft.fft(vXbx)
    fvXby = ft.fft(vXby)
    fvXbz = ft.fft(vXbz)
    output = 2*np.sum(vXbx*Ijx+vXby*Ijy+vXbz*Ijz)
    output -= ft.ifft(fvXbx*np.conj(fjx) + np.conj(fvXbx)*fjx + fvXby*np.conj(fjy) + np.conj(fvXby)*fjy + fvXbz*np.conj(fjz) + np.conj(fvXbz)*fjz)
    
    return output/np.size(output)