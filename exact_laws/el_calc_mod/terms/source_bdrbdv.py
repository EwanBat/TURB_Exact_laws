from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj


class SourceBdrbdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "bx'", "by'", "bz'",
                      "bx", "by", "bz",
                      "divv'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divvP = sp.symbols(("divv'"))

        drbx = rhoP * bxP - rhoNP * bxNP
        drby = rhoP * byP - rhoNP * byNP
        drbz = rhoP * bzP - rhoNP * bzNP
        
        self.expr = (bxNP * drbx + byNP * drby + bzNP * drbz) * divvP

    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, divv, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, bx, by, bz, divv)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, divv)
    
    def calc_fourier(self, rho, bx, by, bz, divv, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, bx, by, bz, divv, traj=traj)

    def variables(self) -> List[str]:
        return ["rho", "b", "divv"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceBdrbdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceBdrbdv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, bx, by, bz, divv,f=njit(SourceBdrbdv().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]

    return (f(rhoP, rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divvP) 
            + f(rhoNP, rhoP, bxNP, byNP, bzNP, bxP, byP, bzP, divvNP))

@njit
def calc_in_point_with_sympy_traj(t, tp, rho, bx, by, bz, divv,f=njit(SourceBdrbdv().fct)):
    rhoP, rhoNP = rho[tp], rho[t]
    bxP, byP, bzP = bx[tp], by[tp], bz[tp]
    bxNP, byNP, bzNP = bx[t], by[t], bz[t]
    divvP, divvNP = divv[tp], divv[t]

    return (f(rhoP, rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divvP)
            + f(rhoNP, rhoP, bxNP, byNP, bzNP, bxP, byP, bzP, divvNP))

def calc_with_fourier(rho, bx, by, bz, divv, traj=False): 
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    #A*dB*C'-A'*dB*C = A*(B'-B)*C'-A'*(B'-B)*C = A*B'*C' + A'B*C - A'*B'*C - A*B*C'
    fbx = transform(bx)
    fby = transform(by)
    fbz = transform(bz)
    frbdx = transform(rho*bx*divv)
    frbdy = transform(rho*by*divv)
    frbdz = transform(rho*bz*divv)
    fd = transform(divv)
    frbb = transform(rho*bx*bx+rho*by*by+rho*bz*bz)
    
    output = inv_transform(frbdx*np.conj(fbx)+frbdy*np.conj(fby)+frbdz*np.conj(fbz)
                     +np.conj(frbdx)*fbx+np.conj(frbdy)*fby+np.conj(frbdz)*fbz
                     -frbb*np.conj(fd)-np.conj(frbb)*fd)
    
    if traj:
        return output/np.size(output,axis=-1)
    return output/np.size(output)


