from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj


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

    def calc(self, vector: List[int], cube_size: List[int], rho, piso, divv, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, rho, piso, divv)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, piso, divv)

    def calc_fourier(self, rho, piso, divv, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, piso, divv, traj=traj)

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

@njit
def calc_in_point_with_sympy_traj(t, tp, rho, piso, divv,f=njit(SourceRdpisodv().fct)):
    rhoP, rhoNP = rho[tp], rho[t]
    pisoP, pisoNP = piso[tp], piso[t]
    divvP, divvNP = divv[tp], divv[t]
    return f(rhoNP,pisoP,pisoNP,divvP) + f(rhoP,pisoNP,pisoP,divvNP)

def calc_with_fourier(rho, piso, divv, traj=False):
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    #A*dB*C' - A'*dB*C = A*B'*C' + A'*B*C - A*B*C' - A'*B'*C
    frp = transform(rho*piso)
    fd = transform(divv)
    fr = transform(rho)
    fpd = transform(piso*divv)
    output = inv_transform(np.conj(fr)*fpd + fr*np.conj(fpd) - np.conj(frp)*fd - frp*np.conj(fd))
    if traj:
        return output/np.size(output,axis=-1)
    return output/np.size(output)


