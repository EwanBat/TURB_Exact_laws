from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj


class SourceDpantr(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("pperp'", "ppar'", "pm'",
                      "bx'", "by'", "bz'",
                      "dxvx'", "dyvx'", "dzvx'",
                      "dxvy'", "dyvy'", "dzvy'",
                      "dxvz'", "dyvz'", "dzvz'",
                      "dxvx", "dyvx", "dzvx",
                      "dxvy", "dyvy", "dzvy",
                      "dxvz", "dyvz", "dzvz"
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        IpperpP, IpparP, IpmP = sp.symbols(("pperp'", "ppar'", "pm'"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        dxvxP, dyvxP, dzvxP = sp.symbols(("dxvx'", "dyvx'", "dzvx'"))
        dxvyP, dyvyP, dzvyP = sp.symbols(("dxvy'", "dyvy'", "dzvy'"))
        dxvzP, dyvzP, dzvzP = sp.symbols(("dxvz'", "dyvz'", "dzvz'"))
        dxvxNP, dyvxNP, dzvxNP = sp.symbols(("dxvx", "dyvx", "dzvx"))
        dxvyNP, dyvyNP, dzvyNP = sp.symbols(("dxvy", "dyvy", "dzvy"))
        dxvzNP, dyvzNP, dzvzNP = sp.symbols(("dxvz", "dyvz", "dzvz"))

        ddxvx = dxvxP - dxvxNP
        ddyvx = dyvxP - dyvxNP
        ddzvx = dzvxP - dzvxNP
        ddxvy = dxvyP - dxvyNP
        ddyvy = dyvyP - dyvyNP
        ddzvy = dzvyP - dzvyNP
        ddxvz = dxvzP - dxvzNP
        ddyvz = dyvzP - dyvzNP
        ddzvz = dzvzP - dzvzNP

        pressP = (IpparP - IpperpP) / (2*IpmP)
        corpressP = (IpperpP - IpparP)/3

        self.expr = (pressP * (IbxP * (IbxP * ddxvx + IbyP * ddxvy + IbzP * ddxvz) 
                              + IbyP * (IbxP * ddyvx + IbyP * ddyvy + IbzP * ddyvz) 
                              + IbzP * (IbxP * ddzvx + IbyP * ddzvy + IbzP * ddzvz))
                     + corpressP * (ddxvx + ddyvy + ddzvz))
        
    def calc(self, vector: List[int], cube_size: List[int],
             Ipperp, Ippar, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz,
             traj=False, **kwarg) -> (float):
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size,
                                      Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)
    
    def calc_fourier(self, Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz, traj=False, **kwarg) -> List:
        return calc_with_fourier(Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz, traj=traj)

    def variables(self) -> List[str]:
        return ["Ipgyr", "Ipm", "gradv", "Ib"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceDpantr()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDpantr().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             Ipperp, Ippar, Ipm,
                             Ibx, Iby, Ibz,
                             dxvx, dyvx, dzvx,
                             dxvy, dyvy, dzvy,
                             dxvz, dyvz, dzvz,  
                             f=njit(SourceDpantr().fct)):
    IpperpP, IpparP, IpmP = Ipperp[ip, jp, kp], Ippar[ip, jp, kp], Ipm[ip, jp, kp]
    IpperpNP, IpparNP, IpmNP = Ipperp[i, j, k], Ippar[i, j, k], Ipm[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    dxvxP, dyvxP, dzvxP = dxvx[ip, jp, kp], dyvx[ip, jp, kp], dzvx[ip, jp, kp]
    dxvyP, dyvyP, dzvyP = dxvy[ip, jp, kp], dyvy[ip, jp, kp], dzvy[ip, jp, kp]
    dxvzP, dyvzP, dzvzP = dxvz[ip, jp, kp], dyvz[ip, jp, kp], dzvz[ip, jp, kp]
    dxvxNP, dyvxNP, dzvxNP = dxvx[i, j, k], dyvx[i, j, k], dzvx[i, j, k]
    dxvyNP, dyvyNP, dzvyNP = dxvy[i, j, k], dyvy[i, j, k], dzvy[i, j, k]
    dxvzNP, dyvzNP, dzvzNP = dxvz[i, j, k], dyvz[i, j, k], dzvz[i, j, k]
    
    return (f(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP)
           + f(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
               dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP
            ))

@njit
def calc_in_point_with_sympy_traj(t, tp, 
                     Ipperp, Ippar, Ipm,
                     Ibx, Iby, Ibz,
                     dxvx, dyvx, dzvx,
                     dxvy, dyvy, dzvy,
                     dxvz, dyvz, dzvz,  
                     f=njit(SourceDpantr().fct)):
    IpperpP, IpparP, IpmP = Ipperp[tp], Ippar[tp], Ipm[tp]
    IpperpNP, IpparNP, IpmNP = Ipperp[t], Ippar[t], Ipm[t]
    IbxP, IbyP, IbzP = Ibx[tp], Iby[tp], Ibz[tp]
    IbxNP, IbyNP, IbzNP = Ibx[t], Iby[t], Ibz[t]
    dxvxP, dyvxP, dzvxP = dxvx[tp], dyvx[tp], dzvx[tp]
    dxvyP, dyvyP, dzvyP = dxvy[tp], dyvy[tp], dzvy[tp]
    dxvzP, dyvzP, dzvzP = dxvz[tp], dyvz[tp], dzvz[tp]
    dxvxNP, dyvxNP, dzvxNP = dxvx[t], dyvx[t], dzvx[t]
    dxvyNP, dyvyNP, dzvyNP = dxvy[t], dyvy[t], dzvy[t]
    dxvzNP, dyvzNP, dzvzNP = dxvz[t], dyvz[t], dzvz[t]

    return (f(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
        dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
        dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP)
        + f(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
        dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP
        ))
                             
def calc_with_fourier(Ipperp, Ippar, Ipm, Ibx, Iby, Ibz, dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, traj=False):
    transform = ft.fft(Ipperp, traj=traj)
    inv_transform = ft.ifft(Ipperp, traj=traj)

    #dA*dB = 2AB - A'B - AB'
    output = 2*np.sum((Ippar - Ipperp) / (2*Ipm) * (Ibx*Ibx*dxvx + Iby*Iby*dyvy + Ibz*Ibz*dzvz
                                                     +Ibx*Iby*(dxvy+dyvx) + Ibx*Ibz*(dxvz+dzvx) + Iby*Ibz*(dzvy+dyvz))
                       + (Ipperp - Ippar)/3 * (dxvx+dyvy+dzvz))
    
    fpbbxx = transform((Ipperp - Ippar)/3 + (Ippar - Ipperp) * Ibx * Ibx / (2*Ipm))
    fpbbxy = transform((Ippar - Ipperp) * Ibx * Iby / (2*Ipm))
    fpbbxz = transform((Ippar - Ipperp) * Ibx * Ibz / (2*Ipm))
    fpbbyy = transform((Ipperp - Ippar)/3 + (Ippar - Ipperp) * Iby * Iby / (2*Ipm))
    fpbbyz = transform((Ippar - Ipperp) * Iby * Ibz / (2*Ipm))
    fpbbzz = transform((Ipperp - Ippar)/3 + (Ippar - Ipperp) * Ibz * Ibz / (2*Ipm))
    
    fdxx = transform(dxvx)
    fdxy = transform(dxvy + dyvx)
    fdxz = transform(dxvz + dzvx)
    fdyy = transform(dyvy)
    fdyz = transform(dzvy + dyvz)
    fdzz = transform(dzvz)
    
    output -= inv_transform(fpbbxx*np.conj(fdxx) + fpbbxy*np.conj(fdxy) + fpbbxz*np.conj(fdxz)
                      + fpbbyy*np.conj(fdyy) + fpbbyz*np.conj(fdyz) + fpbbzz*np.conj(fdzz)
                      + np.conj(fpbbxx)*fdxx + np.conj(fpbbxy)*fdxy + np.conj(fpbbxz)*fdxz
                      + np.conj(fpbbyy)*fdyy + np.conj(fpbbyz)*fdyz + np.conj(fpbbzz)*fdzz) 
    
    if traj:
        return output/np.size(output,axis=-1)
    return output/np.size(output)
    

