from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdpisodv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho", "piso'", "piso", "divv'")
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        rhoNP = sp.symbols(("rho"))
        pisoP, pisoNP = sp.symbols(("piso'", "piso"))
        divvP = sp.symbols(("divv'"))
        
        self.expr = rhoNP * (pisoP - pisoNP) * divvP

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
    return SourceRdpisodv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRdpisodv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, piso, divv,f=njit(SourceRdpisodv().fct)):
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pisoP, pisoNP = piso[ip, jp, kp], piso[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]
    return f(rhoNP,pisoP,pisoNP,divvP) + f(rhoP,pisoNP,pisoP,divvNP)

def calc_with_fourier(rho, piso, divv):
    #A*dB*C' - A'*dB*C = A*B'*C' + A'*B*C - A*B*C' - A'*B'*C
    frp = ft.fft(rho*piso)
    fd = ft.fft(divv)
    fr = ft.fft(rho)
    fpd = ft.fft(piso*divv)
    output = ft.ifft(np.conj(fr)*fpd + fr*np.conj(fpd) - np.conj(frp)*fd - frp*np.conj(fd))
    return output/np.size(output)


