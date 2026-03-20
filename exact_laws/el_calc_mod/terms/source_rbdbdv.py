from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRbdbdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho",
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
        rhoNP = sp.symbols(("rho"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divvP = sp.symbols(("divv'"))

        dbx = bxP - bxNP
        dby = byP - byNP
        dbz = bzP - bzNP
        
        self.expr = rhoNP  * (bxNP * dbx + byNP * dby + bzNP * dbz) * divvP
        
    def calc(self, vector: List[int], cube_size: List[int], rho, bx, by, bz, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, bx, by, bz, divv)
    
    def calc_fourier(self, rho, bx, by, bz, divv, **kwarg) -> List:
        return calc_with_fourier(rho,  bx, by, bz, divv)

    def variables(self) -> List[str]:
        return ["rho", "b", "divv"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceRbdbdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRbdbdv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, bx, by, bz, divv,f=njit(SourceRbdbdv().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]

    return (f(rhoNP, bxP, byP, bzP, bxNP, byNP, bzNP, divvP) 
            + f(rhoP, bxNP, byNP, bzNP, bxP, byP, bzP, divvNP))
    
def calc_with_fourier(rho, bx, by, bz, divv):
    #A*dB*C'-A'*dB*C = A*(B'-B)*C'-A'*(B'-B)*C = A*B'*C' + A'B*C - A'*B'*C - A*B*C'
    frbx = ft.fft(rho*bx)
    frby = ft.fft(rho*by)
    frbz = ft.fft(rho*bz)
    fbdx = ft.fft(bx*divv)
    fbdy = ft.fft(by*divv)
    fbdz = ft.fft(bz*divv)
    fd = ft.fft(divv)
    frbb = ft.fft(rho*bx*bx+rho*by*by+rho*bz*bz)
    
    output = ft.ifft(fbdx*np.conj(frbx)+fbdy*np.conj(frby)+fbdz*np.conj(frbz)
                     +np.conj(fbdx)*frbx+np.conj(fbdy)*frby+np.conj(fbdz)*frbz
                     -frbb*np.conj(fd)-np.conj(frbb)*fd)
    return output/np.size(output)
    
