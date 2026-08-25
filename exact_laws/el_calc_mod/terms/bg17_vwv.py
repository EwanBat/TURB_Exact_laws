from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj

class Bg17Vwv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("vx'", "vy'", "vz'", "vx", "vy", "vz",
                 "wx'", "wy'", "wz'", "wx", "wy", "wz"
                )),
            self.expr,
            "numpy",)
        
    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        wxP, wyP, wzP = sp.symbols(("wx'", "wy'", "wz'"))
        wxNP, wyNP, wzNP = sp.symbols(("wx", "wy", "wz"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP

        vXwxP = vyP * wzP - vzP * wyP  
        vXwyP = vzP * wxP - vxP * wzP  
        vXwzP = vxP * wyP - vyP * wxP  
        vXwxNP = vyNP * wzNP - vzNP * wyNP  
        vXwyNP = vzNP * wxNP - vxNP * wzNP  
        vXwzNP = vxNP * wyNP - vyNP * wxNP

        self.expr = (vXwxP - vXwxNP) * dvx + (vXwyP - vXwyNP) * dvy + (vXwzP - vXwzNP) * dvz
        
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, wx, wy, wz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, vx, vy, vz, wx, wy, wz, traj=traj)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, wx, wy, wz)

    def calc_fourier(self, vx, vy, vz, wx, wy, wz, traj=False, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, wx, wy, wz, traj=traj)

    def variables(self) -> List[str]:
        return ['w','v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return Bg17Vwv()

def print_expr():
    return Bg17Vwv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             wx, wy, wz,
                             f=njit(Bg17Vwv().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    wxP, wyP, wzP = wx[ip, jp, kp], wy[ip, jp, kp], wz[ip, jp, kp]
    wxNP, wyNP, wzNP = wx[i, j, k], wy[i, j, k], wz[i, j, k]
    
    return f(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        wxP, wyP, wzP, wxNP, wyNP, wzNP
    )

@njit
def calc_in_point_with_sympy_traj(tp, t,
                             vx, vy, vz,
                             wx, wy, wz,
                             f=njit(Bg17Vwv().fct)):
    
    vxP, vyP, vzP = vx[tp], vy[tp], vz[tp]
    vxNP, vyNP, vzNP = vx[t], vy[t], vz[t]

    wxP, wyP, wzP = wx[tp], wy[tp], wz[tp]
    wxNP, wyNP, wzNP = wx[t], wy[t], wz[t]

    return f(
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        wxP, wyP, wzP, wxNP, wyNP, wzNP
    )

def calc_with_fourier(vx, vy, vz, wx, wy, wz, traj=False):
    transform = ft.fft(vx, traj=traj)
    inv_transform = ft.ifft(vx, traj=traj)

    fvx = transform(vx)
    fvy = transform(vy)
    fvz = transform(vz)
    
    vXwx = vy * wz - vz * wy
    vXwy = vz * wx - vx * wz
    vXwz = vx * wy - vy * wx
    
    fvXwx = transform(vXwx)
    fvXwy = transform(vXwy)
    fvXwz = transform(vXwz)
    output = 2*np.sum(vXwx*vx+vXwy*vy+vXwz*vz)
    output -= inv_transform(fvXwx*np.conj(fvx) + np.conj(fvXwx)*fvx + fvXwy*np.conj(fvy) + np.conj(fvXwy)*fvy + fvXwz*np.conj(fvz) + np.conj(fvXwz)*fvz)

    if traj:
        return output/np.size(output,axis=-1)
    return output/np.size(output)