from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj

class CorRbb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("rho'","rho",
                 "bx'", "by'", "bz'", "bx", "by", "bz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        
        psbb = bxP*bxNP + byP*byNP + bzP*bzNP

        self.expr = (rhoP+rhoNP)*psbb/2 /2  

    def calc(
        self, vector: List[int], cube_size: List[int], rho, bx, by, bz, traj=False, **kwarg
    ) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, bx, by, bz)
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz)

    def calc_fourier(self, rho, bx, by, bz, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, bx, by, bz, traj=traj)
        
    def variables(self) -> List[str]:
        return ["b", "rho"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return CorRbb()


def print_expr():
    return CorRbb().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho,
                             bx, by, bz, 
                             f=njit(CorRbb().fct)):
    
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    return f(rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP
    )

@njit
def calc_in_point_with_sympy_traj(tp, t,
                                  rho,
                                  bx, by, bz,
                                  f=njit(CorRbb().fct)):
    
    bxP, byP, bzP = bx[tp], by[tp], bz[tp]
    bxNP, byNP, bzNP = bx[t], by[t], bz[t]

    rhoP, rhoNP = rho[tp], rho[t]

    return f(rhoP, rhoNP,
        bxP, byP, bzP, bxNP, byNP, bzNP
    )

def calc_with_fourier(rho, bx, by, bz, traj=False) -> List:

    fbx = ft.fft(bx, traj=traj)
    fby = ft.fft(by, traj=traj)
    fbz = ft.fft(bz, traj=traj)
    frhobx = ft.fft(rho*bx, traj=traj)
    frhoby = ft.fft(rho*by, traj=traj)
    frhobz = ft.fft(rho*bz, traj=traj)
    
    output = ft.ifft(frhobx*np.conj(fbx) + frhoby*np.conj(fby) + frhobz*np.conj(fbz)
                    + np.conj(frhobx)*fbx + np.conj(frhoby)*fby + np.conj(frhobz)*fbz, traj=traj)/4
    
    return output/np.size(output)