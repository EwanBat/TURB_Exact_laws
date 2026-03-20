from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRpisodv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho", "piso'", "divv'")
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        rhoNP = sp.symbols(("rho"))
        pisoP = sp.symbols(("piso'"))
        divvP = sp.symbols(("divv'"))
        
        self.expr = rhoNP * pisoP  * divvP

    def calc(self, vector: List[int], cube_size: List[int], rho, piso, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, piso, divv)

    def calc_fourier(self, rho, piso, divv, **kwarg) -> List:
        return calc_with_fourier(rho, piso, divv)

    def variables(self) -> List[str]:
        return ["rho", "piso", "divv"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceRpisodv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRpisodv().expr


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, piso, divv,f=njit(SourceRpisodv().fct)):
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pisoP, pisoNP = piso[ip, jp, kp], piso[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]
    return f(rhoNP,pisoP,divvP) + f(rhoP,pisoNP,divvNP)

def calc_with_fourier(rho, piso, divv):
    #A*B'*C' + A'*B*C 
    fr = ft.fft(rho)
    fpd = ft.fft(piso*divv)
    output = ft.ifft(np.conj(fr)*fpd + fr*np.conj(fpd))
    return output/np.size(output)
