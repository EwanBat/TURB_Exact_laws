import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from trajectories.derivation_satellite import gradient_1satellite

class GradV2:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradv2'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, ltraj_list: list = None, nbsatellites: int = None):
        inc = 'I' * self.incompressible
        v2 = dic_quant[f"{inc}vx"]*dic_quant[f"{inc}vx"] + dic_quant[f"{inc}vy"]*dic_quant[f"{inc}vy"] + dic_quant[f"{inc}vz"]*dic_quant[f"{inc}vz"]
        
        if traj:
            if nbsatellites == 1:
                gradv2 = gradient_1satellite(
                    np.array([v2]),
                    ltraj_list
                )
            for axisd in ('x', 'y', 'z'):
                ds_name = f"{inc}d{axisd}v2"
                file.create_dataset(
                    ds_name,
                    data = gradv2[0],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )
        else:
            dxv2, dyv2, dzv2 = derivation.grad(v2, 
                dic_param["c"], 
                precision = 4, 
                period = True
            )
            for axisd in ('x', 'y', 'z'):
                ds_name = f"{inc}d{axisd}v2"
                file.create_dataset(
                    ds_name,
                    data = eval(f"d{axisd}v2"),
                    shape = dic_param["N"],
                    dtype = np.float64,
                )      
        
def load(incompressible=False):
    gradv2 = GradV2(incompressible=incompressible)
    return gradv2.name, gradv2
