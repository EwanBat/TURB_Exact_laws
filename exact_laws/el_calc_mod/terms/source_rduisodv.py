from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRduisodv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho", "uiso'", "uiso", "divv'")
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        rhoNP = sp.symbols(("rho"))
        uisoP, uisoNP = sp.symbols(("uiso'", "uiso"))
        divvP = sp.symbols(("divv'"))
        
        self.expr = rhoNP * (uisoP - uisoNP) * divvP

    def calc(self, vector: List[int], cube_size: List[int], rho, uiso, divv, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, uiso, divv)

    def calc_fourier(self, rho, uiso, divv, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, uiso, divv, traj=traj)

    def variables(self) -> List[str]:
        return ["rho", "uiso", "divv"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceRduisodv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRduisodv().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, uiso, divv,f=njit(SourceRduisodv().fct)):
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    uisoP, uisoNP = uiso[ip, jp, kp], uiso[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]
    return f(rhoNP,uisoP,uisoNP,divvP) + f(rhoP,uisoNP,uisoP,divvNP)

def calc_with_fourier(rho, uiso, divv, traj=False):
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    #A*dB*C' - A'*dB*C = A*B'*C' + A'*B*C - A*B*C' - A'*B'*C
    fru = transform(rho*uiso)
    fd = transform(divv)
    fr = transform(rho)
    fud = transform(uiso*divv)
    output = inv_transform(np.conj(fr)*fud + fr*np.conj(fud) - np.conj(fru)*fd - fru*np.conj(fd))
    return output/np.size(output)



