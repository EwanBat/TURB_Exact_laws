from typing import List
from numba import njit
import sympy as sp
import numpy as np

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj

class FluxDjdbdb(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("jx'", "jy'", "jz'",
                 "jx", "jy", "jz",
                 "bx'", "by'", "bz'",
                 "bx", "by", "bz",
                )
        self.fctx = sp.lambdify(
            sp.symbols(quantities),
            self.exprx,
            "numpy",
        )
        self.fcty = sp.lambdify(
            sp.symbols(quantities),
            self.expry,
            "numpy",
        )
        self.fctz = sp.lambdify(
            sp.symbols(quantities),
            self.exprz,
            "numpy",
        )
    
    def set_sympy_expr(self):
        IjxP, IjyP, IjzP = sp.symbols(("jx'", "jy'", "jz'"))
        IjxNP, IjyNP, IjzNP = sp.symbols(("jx", "jy", "jz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))

        dIjx = IjxP - IjxNP
        dIjy = IjyP - IjyNP
        dIjz = IjzP - IjzNP
    
        dIbx = IbxP - IbxNP
        dIby = IbyP - IbyNP
        dIbz = IbzP - IbzNP
    
        self.exprx = (dIjx * dIbx + dIjy * dIby + dIjz * dIbz) * dIbx
        self.expry = (dIjx * dIbx + dIjy * dIby + dIjz * dIbz) * dIby
        self.exprz = (dIjx * dIbx + dIjy * dIby + dIjz * dIbz) * dIbz
        
    def calc(self, vector:List[int], cube_size:List[int], Ijx, Ijy, Ijz, Ibx, Iby, Ibz, traj=False, **kwarg) -> List[float]:
        if traj:
            return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size, Ijx, Ijy, Ijz, Ibx, Iby, Ibz)
        else:
            return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, Ijx, Ijy, Ijz, Ibx, Iby, Ibz)

    def calc_fourier(self, Ijx, Ijy, Ijz, Ibx, Iby, Ibz, traj=False, **kwarg) -> List:
        return calc_with_fourier(Ijx, Ijy, Ijz, Ibx, Iby, Ibz, traj=traj)
    
    def variables(self) -> List[str]:
        return ['Ib','Ij']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

def load():
    return FluxDjdbdb()

def print_expr():
    return FluxDjdbdb().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             Ijx, Ijy, Ijz, 
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDjdbdb().fctx),
                             fy=njit(FluxDjdbdb().fcty),
                             fz=njit(FluxDjdbdb().fctz)):
    
    IjxP, IjyP, IjzP = Ijx[ip, jp, kp], Ijy[ip, jp, kp], Ijz[ip, jp, kp]
    IjxNP, IjyNP, IjzNP = Ijx[i, j, k], Ijy[i, j, k], Ijz[i, j, k]
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    
    outx = fx(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outy = fy(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outz = fz(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    
    return outx, outy, outz

@njit
def calc_in_point_with_sympy_traj(t, tp,
                             Ijx, Ijy, Ijz,
                             Ibx, Iby, Ibz,
                             fx=njit(FluxDjdbdb().fctx),
                             fy=njit(FluxDjdbdb().fcty),
                             fz=njit(FluxDjdbdb().fctz)):
    IjxP, IjyP, IjzP = Ijx[tp], Ijy[tp], Ijz[tp]
    IjxNP, IjyNP, IjzNP = Ijx[t], Ijy[t], Ijz[t]

    IbxP, IbyP, IbzP = Ibx[tp], Iby[tp], Ibz[tp]
    IbxNP, IbyNP, IbzNP = Ibx[t], Iby[t], Ibz[t]
    outx = fx(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outy = fy(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    outz = fz(
        IjxP, IjyP, IjzP, 
        IjxNP, IjyNP, IjzNP, 
        IbxP, IbyP, IbzP, 
        IbxNP, IbyNP, IbzNP
    )
    return outx, outy, outz

def calc_with_fourier(Ijx, Ijy, Ijz, Ibx, Iby, Ibz, traj=False):
    transform = ft.fft(Ijx, traj=traj)
    inv_transform = ft.ifft(Ijx, traj=traj)

    fIjx = transform(Ijx)
    fIjy = transform(Ijy)
    fIjz = transform(Ijz)
    fbx = transform(Ibx)
    fby = transform(Iby)
    fbz = transform(Ibz)
    fbxbz = transform(Ibx*Ibz)
    fIjxbx = transform(Ijx*Ibx)
    fIjyby = transform(Ijy*Iby)
    fIjzbz = transform(Ijz*Ibz)

    fbxby = transform(Ibx*Iby)
    fbxIjy = transform(Ibx*Ijy)
    fbxIjz = transform(Ibx*Ijz)
    fbxbx = transform(Ibx*Ibx)
    flux_x = inv_transform(fbx*np.conj(fIjxbx+fIjyby+fIjzbz) - np.conj(fbx)*(fIjxbx+fIjyby+fIjzbz) 
                        + (fbx*np.conj(fIjxbx)+fby*np.conj(fbxIjy)+fbz*np.conj(fbxIjz))
                        - (np.conj(fbx)*fIjxbx+np.conj(fby)*fbxIjy+np.conj(fbz)*fbxIjz)
                        + (fIjx*np.conj(fbxbx)+fIjy*np.conj(fbxby)+fIjz*np.conj(fbxbz))
                        - (np.conj(fIjx)*fbxbx+np.conj(fIjy)*fbxby+np.conj(fIjz)*fbxbz))
    del(fbxIjy,fbxIjz,fbxbx)
    
    fbybz = transform(Iby*Ibz)
    fbyby = transform(Iby*Iby)
    fIjxby = transform(Ijx*Iby)
    fbyIjz = transform(Iby*Ijz)
    flux_y = inv_transform(fby*np.conj(fIjxbx+fIjyby+fIjzbz) - np.conj(fby)*(fIjxbx+fIjyby+fIjzbz) 
                        + (fbx*np.conj(fIjxby)+fby*np.conj(fIjyby)+fbz*np.conj(fbyIjz))
                        - (np.conj(fbx)*fIjxby+np.conj(fby)*fIjyby+np.conj(fbz)*fbyIjz)
                        + (fIjx*np.conj(fbxby)+fIjy*np.conj(fbyby)+fIjz*np.conj(fbybz))
                        - (np.conj(fIjx)*fbxby+np.conj(fIjy)*fbyby+np.conj(fIjz)*fbybz))
    del(fbyby,fIjxby,fbyIjz,fbxby)
    
    fIjxbz = transform(Ijx*Ibz)
    fIjybz = transform(Ijy*Ibz)
    fbzbz = transform(Ibz*Ibz)
    flux_z = inv_transform(fbz*np.conj(fIjxbx+fIjyby+fIjzbz) - np.conj(fbz)*(fIjxbx+fIjyby+fIjzbz) 
                        + (fbx*np.conj(fIjxbz)+fby*np.conj(fIjybz)+fbz*np.conj(fIjzbz))
                        - (np.conj(fbx)*fIjxbz+np.conj(fby)*fIjybz+np.conj(fbz)*fIjzbz)
                        + (fIjx*np.conj(fbxbz)+fIjy*np.conj(fbybz)+fIjz*np.conj(fbzbz))
                        - (np.conj(fIjx)*fbxbz+np.conj(fIjy)*fbybz+np.conj(fIjz)*fbzbz))
    if traj:
        return [flux_x/np.size(flux_x,axis=-1),flux_y/np.size(flux_y,axis=-1),flux_z/np.size(flux_z,axis=-1)]
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 