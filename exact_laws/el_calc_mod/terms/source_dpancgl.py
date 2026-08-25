from typing import List
from numba import njit
import sympy as sp
from .abstract_term import calc_source_with_numba, calc_source_with_numba_traj
from .source_dpan import SourceDpan, calc_in_point_with_sympy, calc_in_point_with_sympy_traj, calc_with_fourier

class SourceDpanCgl(SourceDpan):
    def __init__(self):
        SourceDpan.__init__(self)
        
    def calc(self, vector: List[int], cube_size: List[int],
             Ipperpcgl, Ipparcgl, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz,
             traj=False, **kwarg) -> (float):
        if traj:
            return calc_source_with_numba_traj(calc_in_point_with_sympy_traj, *vector, *cube_size,
                                      Ipperpcgl, Ipparcgl, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size,
                                      Ipperpcgl, Ipparcgl, Ipm,
                                      Ibx, Iby, Ibz,
                                      dxvx, dyvx, dzvx,
                                      dxvy, dyvy, dzvy,
                                      dxvz, dyvz, dzvz)

    def calc_fourier(self, Ipperpcgl, Ipparcgl, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz, traj=False, **kwarg) -> List:
        return calc_with_fourier(Ipperpcgl, Ipparcgl, Ipm,
             Ibx, Iby, Ibz,
             dxvx, dyvx, dzvx,
             dxvy, dyvy, dzvy,
             dxvz, dyvz, dzvz, traj=traj)

    def variables(self) -> List[str]:
        return ["Ipcgl", "Ipm", "gradv", "Ib"]


def load():
    return SourceDpanCgl()

def print_expr():
    sp.init_printing(use_latex=True)
    return SourceDpanCgl().expr


