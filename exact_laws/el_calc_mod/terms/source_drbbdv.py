from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj


class SourceDrbbdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "bx'", "by'", "bz'",
                      "bx", "by", "bz",
                      "divv'", "divv"
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
        divvP, divvNP = sp.symbols(("divv'", "divv"))

        bPbNP = bxP * bxNP + byP * byNP + bzP * bzNP
        
        self.expr = (rhoP + rhoNP) * bPbNP * (divvP + divvNP)

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
    return SourceDrbbdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDrbbdv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, bx, by, bz, divv,f=njit(SourceDrbbdv().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]

    return (f(rhoP, rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divvP, divvNP) )

@njit
def calc_in_point_with_sympy_traj(t, tp, rho, bx, by, bz, divv,f=njit(SourceDrbbdv().fct)):
    rhoP, rhoNP = rho[tp], rho[t]
    bxP, byP, bzP = bx[tp], by[tp], bz[tp]
    bxNP, byNP, bzNP = bx[t], by[t], bz[t]
    divvP, divvNP = divv[tp], divv[t]

    return (f(rhoP, rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divvP, divvNP))

def calc_with_fourier(rho, bx, by, bz, divv, traj=False): 
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    #(A+A')*B*C'*(D+D') = BC'A'D' + BC'AD + BC'A'D + BC'AD'
    fbx = transform(bx)
    fby = transform(by)
    fbz = transform(bz)
    frbdx = transform(rho*bx*divv)
    frbdy = transform(rho*by*divv)
    frbdz = transform(rho*bz*divv)
    
    output = inv_transform(frbdx*np.conj(fbx)+frbdy*np.conj(fby)+frbdz*np.conj(fbz)
                     +np.conj(frbdx)*fbx+np.conj(frbdy)*fby+np.conj(frbdz)*fbz)

    del(fbx,fby,fbz,frbdx,frbdy,frbdz)
    
    frbx = transform(rho*bx)
    frby = transform(rho*by)
    frbz = transform(rho*bz)
    fbdx = transform(bx*divv)
    fbdy = transform(by*divv)
    fbdz = transform(bz*divv)
    
    output += inv_transform(fbdx*np.conj(frbx)+fbdy*np.conj(frby)+fbdz*np.conj(frbz)
                     +np.conj(fbdx)*frbx+np.conj(fbdy)*frby+np.conj(fbdz)*frbz)
    return output/np.size(output)



