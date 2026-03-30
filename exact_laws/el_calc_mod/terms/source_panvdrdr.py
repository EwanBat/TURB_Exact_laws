from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourcePanvdrdr(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "pperp", "ppar", "pm",
                      "vx", "vy", "vz",
                      "bx", "by", "bz",
                      "dxrho'", "dyrho'", "dzrho'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )

    def set_sympy_expr(self):
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        pperpNP, pparNP, pmNP = sp.symbols(("pperp", "ppar", "pm"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        dxrhoP, dyrhoP, dzrhoP = sp.symbols(("dxrho'", "dyrho'", "dzrho'"))
        
        pressNP = (pparNP - pperpNP) / pmNP

        dualprod = (
            vxNP * bxNP * bxNP * dxrhoP
            + vyNP * bxNP * byNP * dxrhoP
            + vzNP * bxNP * bzNP * dxrhoP
            + vxNP * bxNP * byNP * dyrhoP
            + vyNP * byNP * byNP * dyrhoP
            + vzNP * byNP * bzNP * dyrhoP
            + vxNP * bxNP * bzNP * dzrhoP
            + vyNP * byNP * bzNP * dzrhoP
            + vzNP * bzNP * bzNP * dzrhoP
        )
        
        drho = rhoP - rhoNP

        self.expr = drho * pressNP * dualprod / rhoP 


    def calc(
        self, vector: List[int], cube_size: List[int], rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, **kwarg
    ) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho)

    def calc_fourier(self, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, traj=False, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, traj=traj)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pgyr", "pm", "b"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourcePanvdrdr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourcePanvdrdr().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, f=njit(SourcePanvdrdr().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pperpNP, pparNP, pmNP = pperp[i, j, k], ppar[i, j, k], pm[i, j, k]
    pperpP, pparP, pmP = pperp[ip, jp, kp], ppar[ip, jp, kp], pm[ip, jp, kp]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    dxrhoP, dyrhoP, dzrhoP = dxrho[ip, jp, kp], dyrho[ip, jp, kp], dzrho[ip, jp, kp]
    dxrhoNP, dyrhoNP, dzrhoNP = dxrho[i, j, k], dyrho[i, j, k], dzrho[i, j, k]

    return (f(rhoP, rhoNP, pperpNP, pparNP, pmNP, vxNP, vyNP, vzNP, bxNP, byNP, bzNP, dxrhoP, dyrhoP, dzrhoP) 
            + f(rhoNP, rhoP, pperpP, pparP, pmP, vxP, vyP, vzP, bxP, byP, bzP, dxrhoNP, dyrhoNP, dzrhoNP))

def calc_with_fourier(rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, traj=False):
    transform = ft.fft(rho, traj=traj)
    inv_transform = ft.ifft(rho, traj=traj)

    #dA*B*C'/D' - dA*B'*C/D = A'*B*C'/D' + A*B'*C/D - A*B*C'/D' - A'*B'*C/D
    fpvx = transform((ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * bx)
    fpvy = transform((ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * by)
    fpvz = transform((ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * bz)
    fdx = transform(dxrho)
    fdy = transform(dyrho)
    fdz = transform(dzrho)
    output = inv_transform(fpvx*np.conj(fdx) + fpvy*np.conj(fdy) + fpvz*np.conj(fdz)
                     + np.conj(fpvx)*fdx + np.conj(fpvy)*fdy + np.conj(fpvz)*fdz)
    del(fpvx,fpvy,fpvz,fdx,fdy,fdz)
    frpvx = transform(rho * (ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * bx)
    frpvy = transform(rho * (ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * by)
    frpvz = transform(rho * (ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * bz)
    fdrx = transform(dxrho / rho)
    fdry = transform(dyrho / rho)
    fdrz = transform(dzrho / rho)
    output -= inv_transform(frpvx*np.conj(fdrx) + frpvy*np.conj(fdry) + frpvz*np.conj(fdrz)
                     + np.conj(frpvx)*fdrx + np.conj(frpvy)*fdry + np.conj(frpvz)*fdrz)
    return output/np.size(output)