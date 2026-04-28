import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from trajectories.derivation_satellite import gradient_1satellite

class GradV:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradv'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        inc = 'I' * self.incompressible
        if traj:
            for axisv in ('x', 'y', 'z'):
                if traj_param.get('nbsatellites') == 1:
                    gradv = gradient_1satellite(
                        np.array([dic_quant[f"{inc}v{axisv}"]]),
                        traj_param
                    )
                    for i,axisd in enumerate(('x', 'y', 'z')):
                        ds_name = f"{inc}d{axisd}v{axisv}"
                        file.create_dataset(
                            ds_name,
                            data=gradv[i]
                        )
        else:
            for axisv in ('x', 'y', 'z'):
                dxv, dyv, dzv = derivation.grad(
                    dic_quant[f"{inc}v{axisv}"], 
                    dic_param["c"], 
                    precision = 4, 
                    period = True
                )
                for axisd in ('x', 'y', 'z'):
                    ds_name = f"{inc}d{axisd}v{axisv}"
                    file.create_dataset(
                        ds_name,
                        data = eval(f"d{axisd}v"),
                        shape = dic_param["N"],
                        dtype = np.float64,
                    )      
        
def load(incompressible=False):
    gradv = GradV(incompressible=incompressible)
    return gradv.name, gradv
