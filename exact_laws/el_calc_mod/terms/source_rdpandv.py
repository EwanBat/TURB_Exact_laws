from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba


class SourceRdpandv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("rho", "pperp'", "ppar'", "pm'",
                      "pperp", "ppar", "pm",
                      "bx'", "by'", "bz'",
                      "bx", "by", "bz",
                      "dxvx'", "dxvy'", "dxvz'",
                      "dyvx'", "dyvy'", "dyvz'",
                      "dzvx'", "dzvy'", "dzvz'"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        rhoNP = sp.symbols(("rho"))
        pperpP, pparP, pmP = sp.symbols(("pperp'", "ppar'", "pm'"))
        pperpNP, pparNP, pmNP = sp.symbols(("pperp", "ppar", "pm"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        dxvxP, dxvyP, dxvzP = sp.symbols(("dxvx'", "dxvy'", "dxvz'"))
        dyvxP, dyvyP, dyvzP = sp.symbols(("dyvx'", "dyvy'", "dyvz'"))
        dzvxP, dzvyP, dzvzP = sp.symbols(("dzvx'", "dzvy'", "dzvz'"))
        
        pressP = (pparP - pperpP) / (2*pmP)
        pressNP = (pparNP - pperpNP) / (2*pmNP)

        dpxx = pressP * bxP * bxP - pressNP * bxNP * bxNP
        dpxy = pressP * bxP * byP - pressNP * bxNP * byNP
        dpxz = pressP * bxP * bzP - pressNP * bxNP * bzNP
        dpyy = pressP * byP * byP - pressNP * byNP * byNP
        dpyz = pressP * byP * bzP - pressNP * byNP * bzNP
        dpzz = pressP * bzP * bzP - pressNP * bzNP * bzNP

        self.expr = rhoNP * (dpxx*dxvxP + dpxy*(dyvxP+dxvyP) + dpxz*(dzvxP+dxvzP) 
                             + dpyy*dyvyP + dpyz*(dzvyP+dyvzP) + dpzz*dzvzP)
        

    def calc(self, vector: List[int], cube_size: List[int],
        rho, pperp, ppar, pm, bx, by, bz,
        dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz,
        **kwarg) -> (float):
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
            rho, pperp, ppar, pm, bx, by, bz,
        dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz)
    
    def calc_fourier(self, rho, pperp, ppar, pm, bx, by, bz,
        dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, **kwarg) -> List:
        return calc_with_fourier(rho, pperp, ppar, pm, bx, by, bz,
        dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz)
    
    def variables(self) -> List[str]:
        return ["rho", "pgyr", "pm", "gradv", "b"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceRdpandv()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceRdpandv().expr

@njit
def calc_in_point_with_sympy(
    i, j, k, ip, jp, kp, rho, pperp, ppar, pm, bx, by, bz, dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz,
    f=njit(SourceRdpandv().fct)):
    
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    pperpP, pparP, pmP = pperp[ip, jp, kp], ppar[ip, jp, kp], pm[ip, jp, kp]
    pperpNP, pparNP, pmNP = pperp[i, j, k], ppar[i, j, k], pm[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    dxvxP, dxvyP, dxvzP = dxvx[ip, jp, kp], dxvy[ip, jp, kp], dxvz[ip, jp, kp]
    dyvxP, dyvyP, dyvzP = dyvx[ip, jp, kp], dyvy[ip, jp, kp], dyvz[ip, jp, kp]
    dzvxP, dzvyP, dzvzP = dzvx[ip, jp, kp], dzvy[ip, jp, kp], dzvz[ip, jp, kp]
    dxvxNP, dxvyNP, dxvzNP = dxvx[i, j, k], dxvy[i, j, k], dxvz[i, j, k]
    dyvxNP, dyvyNP, dyvzNP = dyvx[i, j, k], dyvy[i, j, k], dyvz[i, j, k]
    dzvxNP, dzvyNP, dzvzNP = dzvx[i, j, k], dzvy[i, j, k], dzvz[i, j, k]
    

    return (f(rhoNP, pperpP, pparP, pmP, pperpNP, pparNP, pmNP, bxP, byP, bzP, bxNP, byNP, bzNP, dxvxP, dxvyP, dxvzP,
                      dyvxP, dyvyP, dyvzP, dzvxP, dzvyP, dzvzP) 
            + f(rhoP, pperpNP, pparNP, pmNP, pperpP, pparP, pmP, bxNP, byNP, bzNP, bxP, byP, bzP, dxvxNP, dxvyNP, dxvzNP,
                      dyvxNP, dyvyNP, dyvzNP, dzvxNP, dzvyNP, dzvzNP))

def calc_with_fourier(rho, pperp, ppar, pm, bx, by, bz,
        dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz):
    #A*dB*C' - A'*dB*C = A*B'*C' + A'*B*C - A*B*C' - A'*B'*C
    fr = ft.fft(rho)
    fpd = ft.fft((ppar - pperp) / (2*pm) * (bx*bx*dxvx + by*by*dyvy + bz*bz*dzvz 
                                            + bx*by*(dxvy+dyvx) + bx*bz*(dxvz+dzvx) + by*bz*(dyvz+dzvy)))
    frpxx = ft.fft(rho * (ppar - pperp) / (2*pm) * bx * bx)
    frpxy = ft.fft(rho * (ppar - pperp) / (2*pm) * bx * by)
    frpxz = ft.fft(rho * (ppar - pperp) / (2*pm) * bx * bz)
    frpyy = ft.fft(rho * (ppar - pperp) / (2*pm) * by * by)
    frpyz = ft.fft(rho * (ppar - pperp) / (2*pm) * by * bz)
    frpzz = ft.fft(rho * (ppar - pperp) / (2*pm) * bz * bz)
    fdxx = ft.fft(dxvx)
    fdxy = ft.fft(dxvy+dyvx)
    fdxz = ft.fft(dxvz+dzvx)
    fdyy = ft.fft(dyvy)
    fdzz = ft.fft(dzvz)
    fdyz = ft.fft(dzvy+dyvz)
    output = ft.ifft(fr*np.conj(fpd) + np.conj(fr)*fpd
                   - (frpxx*np.conj(fdxx) + frpyy*np.conj(fdyy) + frpzz*np.conj(fdzz)
                      + frpxy*np.conj(fdxy) + frpxz*np.conj(fdxz) + frpyz*np.conj(fdyz))
                   - (fdxx*np.conj(frpxx) + fdyy*np.conj(frpyy) + fdzz*np.conj(frpzz)
                       + fdxy*np.conj(frpxy) + fdxz*np.conj(frpxz) + fdyz*np.conj(frpyz)))
    return output/np.size(output)