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


class SourceDbdbdv(AbstractTerm):
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
    return SourceDbdbdv()


def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDbdbdv().expr

@njit
def calc_in_point_with_sympy_traj(t, tp,
                                  vx, vy, vz,
                                  Ibx, Iby, Ibz,
                                  dxvx, dyvx, dzvx,
                                  dxvy, dyvy, dzvy,
                                  dxvz, dyvz, dzvz,
                                  dxbx, dybx, dzbx,
                                  dxby, dyby, dzby,
                                  dxbz, dybz, dzbz,
                                  f=njit(SourceDbdbdv().fct)):
    vxP, vyP, vzP = vx[:,tp], vy[:,tp], vz[:,tp]
    vxNP, vyNP, vzNP = vx[:,t], vy[:,t], vz[:,t]

    IbxP, IbyP, IbzP = Ibx[:,tp], Iby[:,tp], Ibz[:,tp]
    IbxNP, IbyNP, IbzNP = Ibx[:,t], Iby[:,t], Ibz[:,t]

    dxbxP, dybxP, dzbxP = dxbx[:,tp], dybx[:,tp], dzbx[:,tp]
    dxbyP, dybyP, dzbyP = dxby[:,tp], dyby[:,tp], dzby[:,tp]
    dxbzP, dybzP, dzbzP = dxbz[:,tp], dybz[:,tp], dzbz[:,tp]
    dxbxNP, dybxNP, dzbxNP = dxbx[:,t], dybx[:,t], dzbx[:,t]
    dxbyNP, dybyNP, dzbyNP = dxby[:,t], dyby[:,t], dzby[:,t]
    dxbzNP, dybzNP, dzbzNP = dxbz[:,t], dybz[:,t], dzbz[:,t]

    f_xx = 2*IbxP*dxbxNP*vxP + 2*IbxNP*dxbxP*vxNP
    f_xy = 2*IbyP*dxbyNP*vxP + 2*IbyNP*dxbyP*vxNP
    f_xz = 2*IbzP*dxbzNP*vxP + 2*IbzNP*dxbzP*vxNP
    f_yx = 2*IbxP*dybxNP*vyP + 2*IbxNP*dybxP*vyNP
    f_yy = 2*IbyP*dybyNP*vyP + 2*IbyNP*dybyP*vyNP
    f_yz = 2*IbzP*dybzNP*vyP + 2*IbzNP*dybzP*vyNP
    f_zx = 2*IbxP*dzbxNP*vzP + 2*IbxNP*dzbxP*vzNP
    f_zy = 2*IbyP*dzbyNP*vzP + 2*IbyNP*dzbyP*vzNP
    f_zz = 2*IbzP*dzbzNP*vzP + 2*IbzNP*dzbzP*vzNP

    return f_xx + f_xy + f_xz + f_yx + f_yy + f_yz + f_zx + f_zy + f_zz
