from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba

class ForcV(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify(
            sp.symbols(
                ("rho'","rho",
                 "vx'", "vy'", "vz'", "vx", "vy", "vz",
                 "fx'", "fy'", "fz'", "fx", "fy", "fz",
                )),
            self.expr,
            "numpy",)

    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        fxP, fyP, fzP = sp.symbols(("fx'", "fy'", "fz'"))
        fxNP, fyNP, fzNP = sp.symbols(("fx", "fy", "fz"))

        self.expr =  (rhoNP + rhoP)*(fxP*vxNP + fxNP*vxP + fyP*vyNP + fyNP*vyP + fzP*vzNP + fzNP*vzP)

    def calc(self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, fx, fy, fz, **kwarg
        ) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, fx, fy, fz)
    
    def calc_fourier(self, rho, vx, vy, vz, fx, fy, fz, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, fx, fy, fz)

    def variables(self) -> List[str]:
        return ["f", "v", "rho"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return ForcV()


def print_expr():
    return ForcV().print_expr()


@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             rho,
                             vx, vy, vz, 
                             fx, fy, fz,
                             f=njit(ForcV().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    fxP, fyP, fzP = fx[ip, jp, kp], fy[ip, jp, kp], fz[ip, jp, kp]
    fxNP, fyNP, fzNP = fx[i, j, k], fy[i, j, k], fz[i, j, k]
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    
    return f(rhoP, rhoNP,
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        fxP, fyP, fzP, fxNP, fyNP, fzNP
    )

def calc_with_fourier(rho, vx, vy, vz, fx, fy, fz):
    fvx = ft.fft(vx)
    fvy = ft.fft(vy)
    fvz = ft.fft(vz)
    frhovx = ft.fft(rho*vx)
    frhovy = ft.fft(rho*vy)
    frhovz = ft.fft(rho*vz)
    
    ffx = ft.fft(fx)
    ffy = ft.fft(fy)
    ffz = ft.fft(fz)
    frhofx = ft.fft(rho*fx)
    frhofy = ft.fft(rho*fy)
    frhofz = ft.fft(rho*fz)
    
    output = ft.ifft(frhofx*np.conj(fvx) + frhofy*np.conj(fvy) + frhofz*np.conj(fvz)
                   + frhovx*np.conj(ffx) + frhovy*np.conj(ffy) + frhovz*np.conj(ffz)
                   + np.conj(frhofx)*fvx + np.conj(frhofy)*fvy + np.conj(frhofz)*fvz
                   + np.conj(frhovx)*ffx + np.conj(frhovy)*ffy + np.conj(frhovz)*ffz)
    return output/np.size(output)
    