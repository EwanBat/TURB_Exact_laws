from typing import List
from numba import njit
import sympy as sp
import numpy as np
from scipy.signal import butter, filtfilt

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_source_with_numba, calc_source_with_numba_traj, calc_source_with_numba_traj_split, calc_source_with_numba_traj_filter


class SourceDpan(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.set_sympy_expr_split()
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

        self.fct_xx = sp.lambdify(
            sp.symbols(quantities),
            self.expr_xx,
            "numpy",
        )
        self.fct_xy = sp.lambdify(
            sp.symbols(quantities),
            self.expr_xy,
            "numpy",
        )
        self.fct_xz = sp.lambdify(
            sp.symbols(quantities),
            self.expr_xz,
            "numpy",
        )
        self.fct_yx = sp.lambdify(
            sp.symbols(quantities),
            self.expr_yx,
            "numpy",
        )
        self.fct_yy = sp.lambdify(
            sp.symbols(quantities),
            self.expr_yy,
            "numpy",
        )
        self.fct_yz = sp.lambdify(
            sp.symbols(quantities),
            self.expr_yz,
            "numpy",
        )
        self.fct_zx = sp.lambdify(
            sp.symbols(quantities),
            self.expr_zx,
            "numpy",
        )
        self.fct_zy = sp.lambdify(
            sp.symbols(quantities),
            self.expr_zy,
            "numpy",
        )
        self.fct_zz = sp.lambdify(
            sp.symbols(quantities),
            self.expr_zz,
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

        self.expr = pressP * (IbxP * (IbxP * ddxvx + IbyP * ddxvy + IbzP * ddxvz) + IbyP * (
                IbxP * ddyvx + IbyP * ddyvy + IbzP * ddyvz) + IbzP * (
                                IbxP * ddzvx + IbyP * ddzvy + IbzP * ddzvz)) 

    def set_sympy_expr_split(self):
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

        self.expr_xx = pressP * IbxP * IbxP * ddxvx
        self.expr_xy = pressP * IbxP * IbyP * ddxvy
        self.expr_xz = pressP * IbxP * IbzP * ddxvz
        self.expr_yx = pressP * IbyP * IbxP * ddyvx
        self.expr_yy = pressP * IbyP * IbyP * ddyvy
        self.expr_yz = pressP * IbyP * IbzP * ddyvz
        self.expr_zx = pressP * IbzP * IbxP * ddzvx
        self.expr_zy = pressP * IbzP * IbyP * ddzvy
        self.expr_zz = pressP * IbzP * IbzP * ddzvz

    def calc(self, vector: List[int], cube_size: List[int],
             Ipperp, Ippar, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz, **kwarg) -> (float):
        #return calc_source_with_numba(calc_in_point, *vector, *cube_size,
                                    #   Ipperp, Ippar, Ipm,
                                    #   Ibx, Iby, Ibz,
                                    #   dxvx, dyvx, dzvx,
                                    #   dxvy, dyvy, dzvy,
                                    #   dxvz, dyvz, dzvz)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      Ipperp, Ippar, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)
    
    def calc_incr_traj(self, n_points, n_trajectories, Ipperp, Ippar, Ipm,
                       Ibx, Iby, Ibz,
                       dxvx, dyvx, dzvx,
                       dxvy, dyvy, dzvy,
                       dxvz, dyvz, dzvz, **kwarg):
        return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, n_points, n_trajectories, Ipperp, Ippar, Ipm,
                                           Ibx, Iby, Ibz,
                                           dxvx, dyvx, dzvx,
                                           dxvy, dyvy, dzvy,
                                           dxvz, dyvz, dzvz)
    
    def calc_incr_traj_split(self, n_points, n_trajectories, Ipperp, Ippar, Ipm,
                       Ibx, Iby, Ibz,
                       dxvx, dyvx, dzvx,
                       dxvy, dyvy, dzvy,
                       dxvz, dyvz, dzvz, **kwarg):
        return calc_source_with_numba_traj_split(calc_in_point_with_sympy_traj_split, n_points, n_trajectories, Ipperp, Ippar, Ipm,
                                                 Ibx, Iby, Ibz,
                                                dxvx, dyvx, dzvx,
                                                dxvy, dyvy, dzvy,
                                                dxvz, dyvz, dzvz)
    
    def calc_filter(self, n_points, n_trajectories, fs, Ipperp, Ippar, Ipm,
                    Ibx, Iby, Ibz,
                    dxvx, dyvx, dzvx,
                    dxvy, dyvy, dzvy,
                    dxvz, dyvz, dzvz, **kwarg):
        acc = np.zeros((n_trajectories, n_points))
        order = 4
        for dl in range(n_points):
            if dl > 2:
                wn = fs / dl
            else:
                wn = fs / 3 
            b, a = butter(order, wn, btype='low', fs=fs)
            Ibx = filtfilt(b, a, Ibx, axis=-1)
            Iby = filtfilt(b, a, Iby, axis=-1)
            Ibz = filtfilt(b, a, Ibz, axis=-1)
            Ipperp = filtfilt(b, a, Ipperp, axis=-1)
            Ippar = filtfilt(b, a, Ippar, axis=-1)
            Ipm = filtfilt(b, a, Ipm, axis=-1)
            dxvx = filtfilt(b, a, dxvx, axis=-1)
            dyvx = filtfilt(b, a, dyvx, axis=-1)
            dzvx = filtfilt(b, a, dzvx, axis=-1)
            dxvy = filtfilt(b, a, dxvy, axis=-1)
            dyvy = filtfilt(b, a, dyvy, axis=-1)
            dzvy = filtfilt(b, a, dzvy, axis=-1)
            dxvz = filtfilt(b, a, dxvz, axis=-1)
            dyvz = filtfilt(b, a, dyvz, axis=-1)
            dzvz = filtfilt(b, a, dzvz, axis=-1)
            acc[:,dl] = calc_source_with_numba_traj_filter(calc_in_point_with_sympy_traj, dl, n_points, n_trajectories, Ipperp, Ippar, Ipm,
                                                              Ibx, Iby, Ibz,
                                                                dxvx, dyvx, dzvx,
                                                                dxvy, dyvy, dzvy,
                                                                dxvz, dyvz, dzvz)
        return acc

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

    def variables(self, nbsatellite: int = 1, method=None) -> List[str]:
        return ["Ipgyr", "Ipm", "gradv", "Ib"]
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceDpan()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDpan().expr

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             Ipperp, Ippar, Ipm,
                             Ibx, Iby, Ibz,
                             dxvx, dyvx, dzvx,
                             dxvy, dyvy, dzvy,
                             dxvz, dyvz, dzvz,  
                             f=njit(SourceDpan().fct)):
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
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP))

@njit
def calc_in_point_with_sympy_traj(t, tp,
                     Ipperp, Ippar, Ipm,
                     Ibx, Iby, Ibz,
                     dxvx, dyvx, dzvx,
                     dxvy, dyvy, dzvy,
                     dxvz, dyvz, dzvz,  
                     f=njit(SourceDpan().fct)):
    IpperpP, IpparP, IpmP = Ipperp[:,tp], Ippar[:,tp], Ipm[:,tp]
    IpperpNP, IpparNP, IpmNP = Ipperp[:,t], Ippar[:,t], Ipm[:,t]

    IbxP, IbyP, IbzP = Ibx[:,tp], Iby[:,tp], Ibz[:,tp]
    IbxNP, IbyNP, IbzNP = Ibx[:,t], Iby[:,t], Ibz[:,t]

    dxvxP, dyvxP, dzvxP = dxvx[:,tp], dyvx[:,tp], dzvx[:,tp]
    dxvyP, dyvyP, dzvyP = dxvy[:,tp], dyvy[:,tp], dzvy[:,tp]
    dxvzP, dyvzP, dzvzP = dxvz[:,tp], dyvz[:,tp], dzvz[:,tp]
    
    dxvxNP, dyvxNP, dzvxNP = dxvx[:,t], dyvx[:,t], dzvx[:,t]
    dxvyNP, dyvyNP, dzvyNP = dxvy[:,t], dyvy[:,t], dzvy[:,t]
    dxvzNP, dyvzNP, dzvzNP = dxvz[:,t], dyvz[:,t], dzvz[:,t]

    return (f(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP)
            + f(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP))

@njit
def calc_in_point_with_sympy_traj_split(t, tp,
                     Ipperp, Ippar, Ipm,
                     Ibx, Iby, Ibz,
                     dxvx, dyvx, dzvx,
                     dxvy, dyvy, dzvy,
                     dxvz, dyvz, dzvz,  
                     fxx=njit(SourceDpan().fct_xx),
                     fxy=njit(SourceDpan().fct_xy),
                     fxz=njit(SourceDpan().fct_xz),
                     fyx=njit(SourceDpan().fct_yx),
                     fyy=njit(SourceDpan().fct_yy),
                     fyz=njit(SourceDpan().fct_yz),
                     fzx=njit(SourceDpan().fct_zx),
                     fzy=njit(SourceDpan().fct_zy),
                     fzz=njit(SourceDpan().fct_zz)):
    
    IpperpP, IpparP, IpmP = Ipperp[:,tp], Ippar[:,tp], Ipm[:,tp]
    IpperpNP, IpparNP, IpmNP = Ipperp[:,t], Ippar[:,t], Ipm[:,t]

    IbxP, IbyP, IbzP = Ibx[:,tp], Iby[:,tp], Ibz[:,tp]
    IbxNP, IbyNP, IbzNP = Ibx[:,t], Iby[:,t], Ibz[:,t]

    dxvxP, dyvxP, dzvxP = dxvx[:,tp], dyvx[:,tp], dzvx[:,tp]
    dxvyP, dyvyP, dzvyP = dxvy[:,tp], dyvy[:,tp], dzvy[:,tp]
    dxvzP, dyvzP, dzvzP = dxvz[:,tp], dyvz[:,tp], dzvz[:,tp]
    
    dxvxNP, dyvxNP, dzvxNP = dxvx[:,t], dyvx[:,t], dzvx[:,t]
    dxvyNP, dyvyNP, dzvyNP = dxvy[:,t], dyvy[:,t], dzvy[:,t]
    dxvzNP, dyvzNP, dzvzNP = dxvz[:,t], dyvz[:,t], dzvz[:,t]

    return (fxx(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP)
            + fxx(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fxy(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP)
            + fxy(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fxz(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fxz(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fyx(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fyx(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)),  \
        (fyy(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fyy(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fyz(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fyz(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fzx(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fzx(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fzy(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fzy(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP)), \
        (fzz(IpperpP, IpparP, IpmP, IbxP, IbyP, IbzP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP) 
            + fzz(IpperpNP, IpparNP, IpmNP, IbxNP, IbyNP, IbzNP,
            dxvxNP, dyvxNP, dzvxNP, dxvyNP, dyvyNP, dzvyNP, dxvzNP, dyvzNP, dzvzNP,
            dxvxP, dyvxP, dzvxP, dxvyP, dyvyP, dzvyP, dxvzP, dyvzP, dzvzP))

                             
def calc_with_fourier(Ipperp, Ippar, Ipm, Ibx, Iby, Ibz, dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, traj=False):
    transform = ft.fft(Ipperp, traj=traj)
    inv_transform = ft.ifft(Ipperp, traj=traj)

    #dA*dB = 2AB - A'B - AB'
    fpbbxx = transform((Ippar - Ipperp) / (2*Ipm) * Ibx * Ibx)
    fpbbxy = transform((Ippar - Ipperp) / (2*Ipm) * Ibx * Iby)
    fpbbxz = transform((Ippar - Ipperp) / (2*Ipm) * Ibx * Ibz)
    fpbbyy = transform((Ippar - Ipperp) / (2*Ipm) * Iby * Iby)
    fpbbyz = transform((Ippar - Ipperp) / (2*Ipm) * Iby * Ibz)
    fpbbzz = transform((Ippar - Ipperp) / (2*Ipm) * Ibz * Ibz)
    
    fdxx = transform(dxvx)
    fdxy = transform(dxvy + dyvx)
    fdxz = transform(dxvz + dzvx)
    fdyy = transform(dyvy)
    fdyz = transform(dzvy + dyvz)
    fdzz = transform(dzvz)
    
    output = -inv_transform(fpbbxx*np.conj(fdxx) + fpbbxy*np.conj(fdxy) + fpbbxz*np.conj(fdxz)
                      + fpbbyy*np.conj(fdyy) + fpbbyz*np.conj(fdyz) + fpbbzz*np.conj(fdzz)
                      + np.conj(fpbbxx)*fdxx + np.conj(fpbbxy)*fdxy + np.conj(fpbbxz)*fdxz
                      + np.conj(fpbbyy)*fdyy + np.conj(fpbbyz)*fdyz + np.conj(fpbbzz)*fdzz) 
    
    if traj:
        output = output + 2*np.sum((Ippar - Ipperp) / (2*Ipm) * (Ibx * Ibx * dxvx + Iby * Iby * dyvy + Ibz * Ibz * dzvz
                       +  Ibx * Iby * (dxvy + dyvx) +  Ibx * Ibz * (dxvz + dzvx) +  Iby * Ibz * (dzvy + dyvz)), axis=-1)[:,np.newaxis]

        return output/np.size(output,axis=-1)
    
    output = output + 2*np.sum((Ippar - Ipperp) / (2*Ipm) * (Ibx * Ibx * dxvx + Iby * Iby * dyvy + Ibz * Ibz * dzvz
                       +  Ibx * Iby * (dxvy + dyvx) +  Ibx * Ibz * (dxvz + dzvx) +  Iby * Ibz * (dzvy + dyvz)))
    return output/np.size(output)
    

