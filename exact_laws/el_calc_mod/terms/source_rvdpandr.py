from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRvdpandr(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho'", "rho",
                      "pperp'", "ppar'", "pm'",
                      "pperp", "ppar", "pm",
                      "vx", "vy", "vz",
                      "bx'", "by'", "bz'",
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
        pperpP, pparP, pmP = sp.symbols(("pperp'", "ppar'", "pm'"))
        pperpNP, pparNP, pmNP = sp.symbols(("pperp", "ppar", "pm"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        dxrhoP, dyrhoP, dzrhoP = sp.symbols(("dxrho'", "dyrho'", "dzrho'"))
        
        pressNP = (pparNP - pperpNP) / (2*pmNP)
        pressP = (pparP - pperpP) / (2*pmP)

        dpxx = pressP * bxP * bxP - pressNP * bxNP * bxNP
        dpxy = pressP * bxP * byP - pressNP * bxNP * byNP
        dpxz = pressP * bxP * bzP - pressNP * bxNP * bzNP
        dpyy = pressP * byP * byP - pressNP * byNP * byNP
        dpyz = pressP * byP * bzP - pressNP * byNP * bzNP
        dpzz = pressP * bzP * bzP - pressNP * bzNP * bzNP
    
        dualprod = (
            vxNP * dpxx * dxrhoP
            + vyNP * dpxy * dxrhoP
            + vzNP * dpxz * dxrhoP
            + vxNP * dpxy * dyrhoP
            + vyNP * dpyy * dyrhoP
            + vzNP * dpyz * dyrhoP
            + vxNP * dpxz * dzrhoP
            + vyNP * dpyz * dzrhoP
            + vzNP * dpzz * dzrhoP)
        
        self.expr = rhoNP / rhoP * dualprod  

    def calc(self, vector: List[int], cube_size: List[int],
        rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, **kwarg) -> List[float]:
        return calc_source_with_numba(
            calc_in_point_with_sympy, *vector, *cube_size, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho
        )

    def calc_fourier(self, rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho, **kwarg) -> List:
        return calc_with_fourier(rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ["rho", "gradrho", "v", "pgyr", "pm", "b"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceRvdpandr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRvdpandr().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, rho, vx, vy, vz, pperp, ppar, pm, 
                             bx, by, bz, dxrho, dyrho, dzrho, f=njit(SourceRvdpandr().fct)):

    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pperpNP, pparNP, pmNP = pperp[i, j, k], ppar[i, j, k], pm[i, j, k]
    pperpP, pparP, pmP = pperp[ip, jp, kp], ppar[ip, jp, kp], pm[ip, jp, kp]
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    dxrhoP, dyrhoP, dzrhoP = dxrho[ip, jp, kp], dyrho[ip, jp, kp], dzrho[ip, jp, kp]
    dxrhoNP, dyrhoNP, dzrhoNP = dxrho[i, j, k], dyrho[i, j, k], dzrho[i, j, k]

    return (f(rhoP, rhoNP, pperpP, pparP, pmP, pperpNP, pparNP, pmNP, vxNP, vyNP, vzNP, bxP, byP, bzP, bxNP, byNP, bzNP, dxrhoP, dyrhoP, dzrhoP) 
            + f(rhoNP, rhoP, pperpNP, pparNP, pmNP, pperpP, pparP, pmP, vxP, vyP, vzP, bxNP, byNP, bzNP, bxP, byP, bzP, dxrhoNP, dyrhoNP, dzrhoNP))

def calc_with_fourier(rho, vx, vy, vz, pperp, ppar, pm, bx, by, bz, dxrho, dyrho, dzrho):
    #A*B*dC*D'/E' - A'*B'*dC*D/E = A*B*C'*D'/E' + A'*B'*C*D/E - A*B*C*D'/E' - A'*B'*C'*D/E
    fpdrx = ft.fft((ppar-pperp)/(2*pm) * (dxrho*bx+dyrho*by+dzrho*bz) * bx /rho)
    fpdry = ft.fft((ppar-pperp)/(2*pm) * (dxrho*bx+dyrho*by+dzrho*bz) * by /rho)
    fpdrz = ft.fft((ppar-pperp)/(2*pm) * (dxrho*bx+dyrho*by+dzrho*bz) * bz /rho)
    frvx = ft.fft(rho*vx)
    frvy = ft.fft(rho*vy)
    frvz = ft.fft(rho*vz)
    output = ft.ifft(fpdrx*np.conj(frvx) + fpdry*np.conj(frvy) + fpdrz*np.conj(frvz)
                     + np.conj(fpdrx)*frvx + np.conj(fpdry)*frvy + np.conj(fpdrz)*frvz)
    del(fpdrx,fpdry,fpdrz,frvx,frvy,frvz)
    frpvx = ft.fft(rho * (ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * bx)
    frpvy = ft.fft(rho * (ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * by)
    frpvz = ft.fft(rho * (ppar-pperp)/(2*pm) * (vx*bx+vy*by+vz*bz) * bz)
    fdrx = ft.fft(dxrho / rho)
    fdry = ft.fft(dyrho / rho)
    fdrz = ft.fft(dzrho / rho)
    output -= ft.ifft(frpvx*np.conj(fdrx) + frpvy*np.conj(fdry) + frpvz*np.conj(fdrz)
                     + np.conj(frpvx)*fdrx + np.conj(frpvy)*fdry + np.conj(frpvz)*fdrz)
    return output/np.size(output)