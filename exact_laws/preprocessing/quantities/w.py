import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from trajectories.derivation_satellite import curl_1satellite

class W:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'w'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, ltraj_list: list = None, nbsatellites: int = None):
        inc = 'I' * self.incompressible
        if traj:
            if nbsatellites == 1:
                w = curl_1satellite(
                    np.array([dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]]),
                    ltraj_list
                )
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = w[['x', 'y', 'z'].index(axis)],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )
        else:
            wx, wy, wz = derivation.rot(
                [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]],
                dic_param["c"],
                precision = 4,
                period = True
            )
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = eval(f"w{axis}"),
                    shape = dic_param["N"],
                    dtype = np.float64,
                )      
        


def load(incompressible=False):
    w = W(incompressible=incompressible)
    return w.name, w
