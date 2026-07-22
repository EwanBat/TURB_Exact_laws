from typing import List

from numba import njit
import sympy as sp
import numpy as np
from scipy.signal import butter, filtfilt

from .abstract_term import (
    AbstractTerm,
    calc_source_with_numba,
    calc_source_with_numba_traj,
    calc_source_with_numba_traj_filter,
)


class SourceDvdbdb(AbstractTerm):
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
    return SourceDvdbdb()


def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDvdbdb().expr



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
                                  f=njit(SourceDvdbdb().fct)):
    vxP, vyP, vzP = vx[:,tp], vy[:,tp], vz[:,tp]
    vxNP, vyNP, vzNP = vx[:,t], vy[:,t], vz[:,t]

    IbxP, IbyP, IbzP = bx[:,tp], by[:,tp], bz[:,tp]
    IbxNP, IbyNP, IbzNP = bx[:,t], by[:,t], bz[:,t]

    dxvxP, dyvxP, dzvxP = dxvx[:,tp], dyvx[:,tp], dzvx[:,tp]
    dxvyP, dyvyP, dzvyP = dxvy[:,tp], dyvy[:,tp], dzvy[:,tp]
    dxvzP, dyvzP, dzvzP = dxvz[:,tp], dyvz[:,tp], dzvz[:,tp]
    dxbxP, dybxP, dzbxP = dxbx[:,tp], dybx[:,tp], dzbx[:,tp]
    dxbyP, dybyP, dzbyP = dxby[:,tp], dyby[:,tp], dzby[:,tp]
    dxbzP, dybzP, dzbzP = dxbz[:,tp], dybz[:,tp], dzbz[:,tp]

    dxvxNP, dyvxNP, dzvxNP = dxvx[:,t], dyvx[:,t], dzvx[:,t]
    dxvyNP, dyvyNP, dzvyNP = dxvy[:,t], dyvy[:,t], dzvy[:,t]
    dxvzNP, dyvzNP, dzvzNP = dxvz[:,t], dyvz[:,t], dzvz[:,t]
    dxbxNP, dybxNP, dzbxNP = dxbx[:,t], dybx[:,t], dzbx[:,t]
    dxbyNP, dybyNP, dzbyNP = dxby[:,t], dyby[:,t], dzby[:,t]
    dxbzNP, dybzNP, dzbzNP = dxbz[:,t], dybz[:,t], dzbz[:,t]

    f_xx = dxbxNP*vxP*IbxP + dxvxNP*IbxP*IbxP + IbxNP*IbxNP*dxvxP + vxNP*IbxNP*dxbxP
    f_xy = dxbyNP*vyP*IbxP + dxvyNP*IbyP*IbxP + IbyNP*IbxNP*dxvyP + vyNP*IbxNP*dxbyP
    f_xz = dxbzNP*vzP*IbxP + dxvzNP*IbzP*IbxP + IbzNP*IbxNP*dxvzP + vzNP*IbxNP*dxbzP
    f_yx = dybxNP*vxP*IbyP + dyvxNP*IbxP*IbyP + IbxNP*IbyNP*dyvxP + vxNP*IbyNP*dybxP
    f_yy = dybyNP*vyP*IbyP + dyvyNP*IbyP*IbyP + IbyNP*IbyNP*dyvyP + vyNP*IbyNP*dybyP
    f_yz = dybzNP*vzP*IbyP + dyvzNP*IbzP*IbyP + IbzNP*IbyNP*dyvzP + vzNP*IbyNP*dybzP
    f_zx = dzbxNP*vxP*IbzP + dzvxNP*IbxP*IbzP + IbxNP*IbzNP*dzvxP + vxNP*IbzNP*dzbxP
    f_zy = dzbyNP*vyP*IbzP + dzvyNP*IbyP*IbzP + IbyNP*IbzNP*dzvyP + vyNP*IbzNP*dzbyP
    f_zz = dzbzNP*vzP*IbzP + dzvzNP*IbzP*IbzP + IbzNP*IbzNP*dzvzP + vzNP*IbzNP*dzbzP

    return f_xx + f_xy + f_xz + f_yx + f_yy + f_yz + f_zx + f_zy + f_zz
