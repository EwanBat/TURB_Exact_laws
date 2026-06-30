from typing import List
from numba import njit
import sympy as sp
import numpy as np
from scipy.signal import filtfilt, butter

from ...mathematical_tools import fourier_transform as ft
from .abstract_term import AbstractTerm, calc_flux_with_numba, calc_flux_with_numba_traj, calc_flux_with_numba_traj_filter

class FluxDvdvdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'",
                 "vx", "vy", "vz"
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
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))

        dvx = vxP - vxNP
        dvy = vyP - vyNP
        dvz = vzP - vzNP
    
        self.exprx = (dvx * dvx + dvy * dvy + dvz * dvz) * dvx
        self.expry = (dvx * dvx + dvy * dvy + dvz * dvz) * dvy
        self.exprz = (dvx * dvx + dvy * dvy + dvz * dvz) * dvz
    
    def calc(self, vector:List[int], cube_size:List[int], vx, vy, vz, **kwarg) -> List[float]:
        return calc_flux_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz)
    
    def calc_incr_traj(self, n_points, n_trajectories, vx, vy, vz, **kwarg):
        return calc_flux_with_numba_traj(calc_in_point_with_sympy_traj, n_points, n_trajectories, vx, vy, vz)

    def calc_filter(self, n_points, n_trajectories, fs, vx, vy, vz, **kwarg):
        acc = np.zeros((3, n_trajectories, n_points))
        order = 0
        for dl in range(n_points):
            if dl // 25 > order:
                order = dl // 25
                b, a = butter(order, 2*np.pi / (dl * 1/fs), btype='low', fs=fs)
                vx = filtfilt(b, a, vx, axis=-1)
                vy = filtfilt(b, a, vy, axis=-1)
                vz = filtfilt(b, a, vz, axis=-1)
            acc[:, :, dl] = calc_flux_with_numba_traj_filter(calc_in_point_with_sympy_traj, dl, n_points, n_trajectories, vx, vy, vz)
        return acc

    def calc_fourier(self, vx, vy, vz, traj=False, **kwarg) -> List:
        return calc_with_fourier(vx, vy, vz, traj=traj)

    def variables(self, nbsatellite=1, method=None) -> List[str]:
        if nbsatellite == 4 and method == "fourier":
            return ['v', 'gradv']
        else:
            return ['v']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.exprx, self.expry, self.exprz

    def calc_with_fourier_4sat(self, vx, vy, vz, dxvx, dyvx, dzvx, dxvy, dyvy, dzvy, dxvz, dyvz, dzvz, traj=True, **kwarg) -> np.ndarray:
        transform = ft.fft(vx, traj=traj)
        inv_transform = ft.ifft(vx, traj=traj)

        fvxvx = transform(vx*vx)
        fvyvy = transform(vy*vy)
        fvzvz = transform(vz*vz)
        fvxvy = transform(vx*vy)
        fvxvz = transform(vx*vz)
        fvyvz = transform(vy*vz)
        fvzvz = transform(vz*vz)

        fdxvx = transform(dxvx)
        fdyvx = transform(dyvx)
        fdzvx = transform(dzvx)
        fdxvy = transform(dxvy)
        fdyvy = transform(dyvy)
        fdzvy = transform(dzvy)
        fdxvz = transform(dxvz)
        fdyvz = transform(dyvz)
        fdzvz = transform(dzvz)

        flux_xx = inv_transform(fvxvx*np.conj(fdxvx) + np.conj(fvxvx)*fdxvx + 2*(fvxvx*np.conj(fdxvx) + np.conj(fvxvx)*fdxvx))
        flux_xy = inv_transform(fvyvy*np.conj(fdxvx) + np.conj(fvyvy)*fdxvx + 2*(fvxvy*np.conj(fdxvy) + np.conj(fvxvy)*fdxvy))
        flux_xz = inv_transform(fvzvz*np.conj(fdxvx) + np.conj(fvzvz)*fdxvx + 2*(fvxvz*np.conj(fdxvz) + np.conj(fvxvz)*fdxvz))
        
        flux_yx = inv_transform(fvxvx*np.conj(fdyvy) + np.conj(fvxvx)*fdyvy + 2*(fvxvy*np.conj(fdyvx) + np.conj(fvxvy)*fdyvx))
        flux_yy = inv_transform(fvyvy*np.conj(fdyvy) + np.conj(fvyvy)*fdyvy + 2*(fvyvy*np.conj(fdyvy) + np.conj(fvyvy)*fdyvy))
        flux_yz = inv_transform(fvzvz*np.conj(fdyvy) + np.conj(fvzvz)*fdyvy + 2*(fvyvz*np.conj(fdyvz) + np.conj(fvyvz)*fdyvz))

        flux_zx = inv_transform(fvxvx*np.conj(fdzvz) + np.conj(fvxvx)*fdzvz + 2*(fvxvz*np.conj(fdzvx) + np.conj(fvxvz)*fdzvx))
        flux_zy = inv_transform(fvyvy*np.conj(fdzvz) + np.conj(fvyvy)*fdzvz + 2*(fvyvz*np.conj(fdzvy) + np.conj(fvyvz)*fdzvy))
        flux_zz = inv_transform(fvzvz*np.conj(fdzvz) + np.conj(fvzvz)*fdzvz + 2*(fvzvz*np.conj(fdzvz) + np.conj(fvzvz)*fdzvz))

        return (flux_xx + flux_xy + flux_xz + flux_yx + flux_yy + flux_yz + flux_zx + flux_zy + flux_zz) / np.size(flux_xx, axis=-1)

def load():
    return FluxDvdvdv()

def print_expr():
    sp.init_printing(use_latex=True)
    return FluxDvdvdv().exprx, FluxDvdvdv().expry, FluxDvdvdv().exprz

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz,  
                             fx=njit(FluxDvdvdv().fctx),
                             fy=njit(FluxDvdvdv().fcty),
                             fz=njit(FluxDvdvdv().fctz)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    
    outx = fx(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    outy = fy(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    outz = fz(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    
    return outx, outy, outz

@njit
def calc_in_point_with_sympy_traj(t, tp, vx, vy, vz,
                                  fx=njit(FluxDvdvdv().fctx),
                                  fy=njit(FluxDvdvdv().fcty),
                                  fz=njit(FluxDvdvdv().fctz)):
    vxP, vyP, vzP = vx[:,tp], vy[:,tp], vz[:,tp]
    vxNP, vyNP, vzNP = vx[:,t], vy[:,t], vz[:,t]

    outx = fx(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    outy = fy(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    outz = fz(vxP, vyP, vzP, vxNP, vyNP, vzNP)
    return outx, outy, outz

@njit
def calc_in_point(i, j, k, ip, jp, kp, vx, vy, vz):
    
    dvx = vx[ip,jp,kp] - vx[i,j,k]
    dvy = vy[ip,jp,kp] - vy[i,j,k]
    dvz = vz[ip,jp,kp] - vz[i,j,k]
    
    fx = (dvx * dvx + dvy * dvy + dvz * dvz) * dvx
    fy = (dvx * dvx + dvy * dvy + dvz * dvz) * dvy
    fz = (dvx * dvx + dvy * dvy + dvz * dvz) * dvz
    
    return fx, fy, fz

def calc_with_fourier(vx, vy, vz, traj=False):
    transform = ft.fft(vx, traj=traj)
    inv_transform = ft.ifft(vx, traj=traj)

    fvx = transform(vx)
    fvy = transform(vy)
    fvz = transform(vz)
    fvxvx = transform(vx*vx)
    fvyvy = transform(vy*vy)
    fvzvz = transform(vz*vz)
    fvxvy = transform(vx*vy)
    fvxvz = transform(vx*vz)
    flux_x = inv_transform(fvx*np.conj(fvxvx+fvyvy+fvzvz) - np.conj(fvx)*(fvxvx+fvyvy+fvzvz) 
                        + 2*(fvx*np.conj(fvxvx)+fvy*np.conj(fvxvy)+fvz*np.conj(fvxvz))
                        - 2*(np.conj(fvx)*fvxvx+np.conj(fvy)*fvxvy+np.conj(fvz)*fvxvz))
    fvyvz = transform(vy*vz)
    flux_y = inv_transform(fvy*np.conj(fvxvx+fvyvy+fvzvz) - np.conj(fvy)*(fvxvx+fvyvy+fvzvz) 
                        + 2*(fvx*np.conj(fvxvy)+fvy*np.conj(fvyvy)+fvz*np.conj(fvyvz))
                        - 2*(np.conj(fvx)*fvxvy+np.conj(fvy)*fvyvy+np.conj(fvz)*fvyvz))
    del(fvxvy)
    flux_z = inv_transform(fvz*np.conj(fvxvx+fvyvy+fvzvz) - np.conj(fvz)*(fvxvx+fvyvy+fvzvz) 
                        + 2*(fvx*np.conj(fvxvz)+fvy*np.conj(fvyvz)+fvz*np.conj(fvzvz))
                        - 2*(np.conj(fvx)*fvxvz+np.conj(fvy)*fvyvz+np.conj(fvz)*fvzvz))
    
    if traj:
        return [flux_x/np.size(flux_x,axis=-1),flux_y/np.size(flux_y,axis=-1),flux_z/np.size(flux_z,axis=-1)]
    return [flux_x/np.size(flux_x),flux_y/np.size(flux_y),flux_z/np.size(flux_z)] 

