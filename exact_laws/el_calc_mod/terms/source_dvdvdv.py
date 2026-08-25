from typing import List

from numba import njit
import sympy as sp
import numpy as np
from scipy.signal import butter, filtfilt

from .abstract_term import (
    AbstractTerm,
    calc_source_with_numba_traj,
    calc_source_with_numba_traj_filter,
)


class SourceDvdvdv(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        self.fct = sp.lambdify((), self.expr, "numpy")

    def set_sympy_expr(self):
        self.expr = sp.Integer(0)

    def calc_incr_traj(self, n_points, n_trajectories,
                       vx, vy, vz,
                       bx, by, bz,
                       dxvx, dyvx, dzvx,
                       dxvy, dyvy, dzvy,
                       dxvz, dyvz, dzvz,
                       dxbx, dybx, dzbx,
                       dxby, dyby, dzby,
                       dxbz, dybz, dzbz,
                       **kwarg):
        return calc_source_with_numba_traj(
            calc_in_point_with_sympy_traj,
            n_points,
            n_trajectories,
            vx, vy, vz,
            bx, by, bz,
            dxvx, dyvx, dzvx,
            dxvy, dyvy, dzvy,
            dxvz, dyvz, dzvz,
            dxbx, dybx, dzbx,
            dxby, dyby, dzby,
            dxbz, dybz, dzbz,
        )

    def calc_filter(self, n_points, n_trajectories, fs,
                    vx, vy, vz,
                    bx, by, bz,
                    dxvx, dyvx, dzvx,
                    dxvy, dyvy, dzvy,
                    dxvz, dyvz, dzvz,
                    dxbx, dybx, dzbx,
                    dxby, dyby, dzby,
                    dxbz, dybz, dzbz,
                    **kwarg):
        acc = np.zeros((n_trajectories, n_points))
        order = 4
        for dl in range(n_points):
            if dl > 2:
                wn = fs / dl
            else:
                wn = fs / 3
            b, a = butter(order, wn, btype='low', fs=fs)
            vx = filtfilt(b, a, vx, axis=-1)
            vy = filtfilt(b, a, vy, axis=-1)
            vz = filtfilt(b, a, vz, axis=-1)
            bx = filtfilt(b, a, bx, axis=-1)
            by = filtfilt(b, a, by, axis=-1)
            bz = filtfilt(b, a, bz, axis=-1)
            dxvx = filtfilt(b, a, dxvx, axis=-1)
            dyvx = filtfilt(b, a, dyvx, axis=-1)
            dzvx = filtfilt(b, a, dzvx, axis=-1)
            dxvy = filtfilt(b, a, dxvy, axis=-1)
            dyvy = filtfilt(b, a, dyvy, axis=-1)
            dzvy = filtfilt(b, a, dzvy, axis=-1)
            dxvz = filtfilt(b, a, dxvz, axis=-1)
            dyvz = filtfilt(b, a, dyvz, axis=-1)
            dzvz = filtfilt(b, a, dzvz, axis=-1)
            dxbx = filtfilt(b, a, dxbx, axis=-1)
            dybx = filtfilt(b, a, dybx, axis=-1)
            dzbx = filtfilt(b, a, dzbx, axis=-1)
            dxby = filtfilt(b, a, dxby, axis=-1)
            dyby = filtfilt(b, a, dyby, axis=-1)
            dzby = filtfilt(b, a, dzby, axis=-1)
            dxbz = filtfilt(b, a, dxbz, axis=-1)
            dybz = filtfilt(b, a, dybz, axis=-1)
            dzbz = filtfilt(b, a, dzbz, axis=-1)
            acc[:, dl] = calc_source_with_numba_traj_filter(
                calc_in_point_with_sympy_traj,
                dl,
                n_points,
                n_trajectories,
                vx, vy, vz,
                bx, by, bz,
                dxvx, dyvx, dzvx,
                dxvy, dyvy, dzvy,
                dxvz, dyvz, dzvz,
                dxbx, dybx, dzbx,
                dxby, dyby, dzby,
                dxbz, dybz, dzbz,
            )
        return acc

    def variables(self, nbsatellite=1, method=None) -> List[str]:
        return ["v", "b", "gradv", "gradb"]

    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr


def load():
    return SourceDvdvdv()


def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDvdvdv().expr

@njit
def calc_in_point_with_sympy_traj(t, tp,
                                  vx, vy, vz,
                                  bx, by, bz,
                                  dxvx, dyvx, dzvx,
                                  dxvy, dyvy, dzvy,
                                  dxvz, dyvz, dzvz,
                                  dxbx, dybx, dzbx,
                                  dxby, dyby, dzby,
                                  dxbz, dybz, dzbz,
                                  f=njit(SourceDvdvdv().fct)):
    vxP, vyP, vzP = vx[:,tp], vy[:,tp], vz[:,tp]
    vxNP, vyNP, vzNP = vx[:,t], vy[:,t], vz[:,t]

    dxvxP, dyvxP, dzvxP = dxvx[:,tp], dyvx[:,tp], dzvx[:,tp]
    dxvyP, dyvyP, dzvyP = dxvy[:,tp], dyvy[:,tp], dzvy[:,tp]
    dxvzP, dyvzP, dzvzP = dxvz[:,tp], dyvz[:,tp], dzvz[:,tp]
    dxvxNP, dyvxNP, dzvxNP = dxvx[:,t], dyvx[:,t], dzvx[:,t]
    dxvyNP, dyvyNP, dzvyNP = dxvy[:,t], dyvy[:,t], dzvy[:,t]
    dxvzNP, dyvzNP, dzvzNP = dxvz[:,t], dyvz[:,t], dzvz[:,t]

    f_xx = 2*vxP*vxP*dxvxNP + 2*dxvxP*vxNP*vxNP
    f_xy = 2*vyP*vxP*dxvyNP + 2*dxvyP*vxNP*vyNP
    f_xz = 2*vzP*vxP*dxvzNP + 2*dxvzP*vxNP*vzNP
    f_yx = 2*vxP*vyP*dyvxNP + 2*dyvxP*vyNP*vxNP
    f_yy = 2*vyP*vyP*dyvyNP + 2*dyvyP*vyNP*vyNP
    f_yz = 2*vzP*vyP*dyvzNP + 2*dyvzP*vyNP*vzNP
    f_zx = 2*vxP*vzP*dzvxNP + 2*dzvxP*vzNP*vxNP
    f_zy = 2*vyP*vzP*dzvyNP + 2*dzvyP*vzNP*vyNP
    f_zz = 2*vzP*vzP*dzvzNP + 2*dzvzP*vzNP*vzNP

    return f_xx + f_xy + f_xz + f_yx + f_yy + f_yz + f_zx + f_zy + f_zz
