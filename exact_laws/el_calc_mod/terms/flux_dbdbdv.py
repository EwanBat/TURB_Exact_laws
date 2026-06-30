from typing import List
from numba import njit
import sympy as sp
import numpy as np
from scipy.signal import filtfilt, butter

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj, calc_flux_with_numba_traj_filter

class FluxDbdbdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "vx", "vy", "vz",
                 "bx'", "by'", "bz'", "bx", "by", "bz")
        
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
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        IbxP, IbyP, IbzP = sp.symbols(("bx'", "by'", "bz'"))
        IbxNP, IbyNP, IbzNP = sp.symbols(("bx", "by", "bz"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
    
        dIbx = IbxP - IbxNP
        dIby = IbyP - IbyNP
        dIbz = IbzP - IbzNP
    
        self.exprx = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvx
        self.expry = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvy
        self.exprz = (dIbx * dIbx + dIby * dIby + dIbz * dIbz) * dvz
        
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, Ibx, Iby, Ibz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, Ibx, Iby, Ibz)
        
    def calc_incr_traj(self, n_points, n_trajectories, vx, vy, vz, Ibx, Iby, Ibz, **kwarg):
        return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, n_points, n_trajectories, vx, vy, vz, Ibx, Iby, Ibz)

    def calc_filter(self, n_points, n_trajectories, fs, vx, vy, vz, Ibx, Iby, Ibz, **kwarg):
        acc = np.zeros((3, n_trajectories, n_points))
        order = 0
        for dl in range(n_points):
            if dl // 25 > order:
                order = dl // 25
                b, a = butter(order, 2*np.pi / (dl * 1/fs), btype='low', fs=fs)
                vx = filtfilt(b, a, vx, axis=-1)
                vy = filtfilt(b, a, vy, axis=-1)
                vz = filtfilt(b, a, vz, axis=-1)
                Ibx = filtfilt(b, a, Ibx, axis=-1)
                Iby = filtfilt(b, a, Iby, axis=-1)
                Ibz = filtfilt(b, a, Ibz, axis=-1)
            acc[:, :, dl] = calc_flux_with_numba_traj_filter(calc_in_point_with_sympy_traj, dl, n_points, n_trajectories, vx, vy, vz, Ibx, Iby, Ibz)
        return acc

    def calc_fourier(self, vx, vy, vz, Ibx, Iby, Ibz, traj=False,**kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz, traj=traj)
    
    def variables(self, nbsatellite=1, method=None) -> List[str]:
        if nbsatellite == 4 and method == "fourier":
            return ['v', 'Ib', 'Igradb', 'gradv']
        else:
            return ['v', 'Ib']

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

    def calc_with_fourier_4sat(self, vx, vy, vz, Ibx, Iby, Ibz, 
                            Idxbx, Idybx, Idzbx, Idxby, Idyby, Idzby, Idxbz, Idybz, Idzbz,
                            dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, traj=True, **kwarg) -> np.ndarray:
        
        transform = ft.fft(vx, traj=traj)
        inv_transform = ft.ifft(vx, traj=traj)

        fbxbx = transform(Ibx*Ibx)
        fbyby = transform(Iby*Iby)
        fbzbz = transform(Ibz*Ibz)

        fbxvx = transform(Ibx*vx)
        fbyvy = transform(Iby*vy)
        fbzvz = transform(Ibz*vz)
        fbxvy = transform(Ibx*vy)
        fbyvx = transform(Iby*vx)
        fbzvx = transform(Ibz*vx)
        fbxvz = transform(Ibx*vz)
        fbyvz = transform(Iby*vz)
        fbzvy = transform(Ibz*vy)

        fdxvx = transform(dxvx)
        fdyvy = transform(dyvy)
        fdzvz = transform(dzvz)

        fdxbx = transform(Idxbx)
        fdxby = transform(Idxby)
        fdxbz = transform(Idxbz)
        fdybx = transform(Idybx)
        fdyby = transform(Idyby)
        fdybz = transform(Idybz)
        fdzbx = transform(Idzbx)
        fdzby = transform(Idzby)
        fdzbz = transform(Idzbz)

        flux_xx = inv_transform(fbxbx*np.conj(fdxvx) + np.conj(fbxbx)*fdxvx + 2*(fbxvx*np.conj(fdxbx) + np.conj(fbxvx)*fdxbx))
        flux_xy = inv_transform(fbyvy*np.conj(fdxvx) + np.conj(fbyvy)*fdxvx + 2*(fbyvx*np.conj(fdxby) + np.conj(fbyvx)*fdxby))
        flux_xz = inv_transform(fbzbz*np.conj(fdxvx) + np.conj(fbzbz)*fdxvx + 2*(fbzvx*np.conj(fdxbz) + np.conj(fbzvx)*fdxbz))

        flux_yx = inv_transform(fbxbx*np.conj(fdyvy) + np.conj(fbxbx)*fdyvy + 2*(fbxvy*np.conj(fdybx) + np.conj(fbxvy)*fdybx))
        flux_yy = inv_transform(fbyby*np.conj(fdyvy) + np.conj(fbyby)*fdyvy + 2*(fbyvy*np.conj(fdyby) + np.conj(fbyvy)*fdyby))
        flux_yz = inv_transform(fbzbz*np.conj(fdyvy) + np.conj(fbzbz)*fdyvy + 2*(fbzvy*np.conj(fdybz) + np.conj(fbzvy)*fdybz))

        flux_zx = inv_transform(fbxbx*np.conj(fdzvz) + np.conj(fbxbx)*fdzvz + 2*(fbxvz*np.conj(fdzbx) + np.conj(fbxvz)*fdzbx))
        flux_zy = inv_transform(fbyby*np.conj(fdzvz) + np.conj(fbyby)*fdzvz + 2*(fbyvz*np.conj(fdzby) + np.conj(fbyvz)*fdzby))
        flux_zz = inv_transform(fbzbz*np.conj(fdzvz) + np.conj(fbzbz)*fdzvz + 2*(fbzvz*np.conj(fdzbz) + np.conj(fbzvz)*fdzbz))

        return (flux_xx + flux_xy + flux_xz + flux_yx + flux_yy + flux_yz + flux_zx + flux_zy + flux_zz) / np.size(flux_xx, axis=-1)

def load():
    return FluxDbdbdv()

def print_expr():
    return FluxDbdbdv().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, 
                             Ibx, Iby, Ibz,  
                             fx=njit(FluxDbdbdv().fctx),
                             fy=njit(FluxDbdbdv().fcty),
                             fz=njit(FluxDbdbdv().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    IbxP, IbyP, IbzP = Ibx[ip, jp, kp], Iby[ip, jp, kp], Ibz[ip, jp, kp]
    IbxNP, IbyNP, IbzNP = Ibx[i, j, k], Iby[i, j, k], Ibz[i, j, k]
    outx = fx(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outy = fy(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outz = fz(
        vxP, vyP, vzP, vxNP, vyNP, vzNP, 
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    return outx, outy, outz

@njit
def calc_in_point_with_sympy_traj(t, tp,
                             vx, vy, vz,
                             Ibx, Iby, Ibz,
                             fx=njit(FluxDbdbdv().fctx),
                             fy=njit(FluxDbdbdv().fcty),
                             fz=njit(FluxDbdbdv().fctz)):
    vxP, vyP, vzP = vx[:,tp], vy[:,tp], vz[:,tp]
    vxNP, vyNP, vzNP = vx[:,t], vy[:,t], vz[:,t]

    IbxP, IbyP, IbzP = Ibx[:,tp], Iby[:,tp], Ibz[:,tp]
    IbxNP, IbyNP, IbzNP = Ibx[:,t], Iby[:,t], Ibz[:,t]

    outx = fx(
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)
    
    outy = fy(
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)

    outz = fz(
        vxP, vyP, vzP, vxNP, vyNP, vzNP,
        IbxP, IbyP, IbzP, IbxNP, IbyNP, IbzNP)

    return outx, outy, outz

def calc_with_fourier(vx, vy, vz, Ibx, Iby, Ibz, traj=False):    
    transform = ft.fft(vx, traj=traj)
    inv_transform = ft.ifft(vx, traj=traj)

    fbx = transform(Ibx)
    fby = transform(Iby)
    fbz = transform(Ibz)
    fbxbx = transform(Ibx*Ibx)
    fbyby = transform(Iby*Iby)
    fbzbz = transform(Ibz*Ibz)

    fvx = transform(vx)
    fvxbx = transform(vx*Ibx)
    fvxby = transform(vx*Iby)
    fvxbz = transform(vx*Ibz)
    flux_x = inv_transform(fvx*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fvx)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fvxbx)+fby*np.conj(fvxby)+fbz*np.conj(fvxbz))
                        - 2*(np.conj(fbx)*fvxbx+np.conj(fby)*fvxby+np.conj(fbz)*fvxbz))
    del(fvxbx,fvxby,fvxbz,fvx)
    
    fvy = transform(vy)
    fbxvy = transform(Ibx*vy)
    fvyby = transform(vy*Iby)
    fvybz = transform(vy*Ibz)
    flux_y = inv_transform(fvy*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fvy)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fbxvy)+fby*np.conj(fvyby)+fbz*np.conj(fvybz))
                        - 2*(np.conj(fbx)*fbxvy+np.conj(fby)*fvyby+np.conj(fbz)*fvybz))
    del(fbxvy,fvyby,fvybz,fvy)
    
    fvz = transform(vz)
    fbxvz = transform(Ibx*vz)
    fbyvz = transform(Iby*vz)
    fvzbz = transform(vz*Ibz)
    flux_z = inv_transform(fvz*np.conj(fbxbx+fbyby+fbzbz) - np.conj(fvz)*(fbxbx+fbyby+fbzbz) 
                        + 2*(fbx*np.conj(fbxvz)+fby*np.conj(fbyvz)+fbz*np.conj(fvzbz))
                        - 2*(np.conj(fbx)*fbxvz+np.conj(fby)*fbyvz+np.conj(fbz)*fvzbz))
    
    if traj:
        return [flux_x/np.size(flux_x,axis=-1),flux_y/np.size(flux_y,axis=-1),flux_z/np.size(flux_z,axis=-1)]
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 