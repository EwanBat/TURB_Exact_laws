import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from trajectories.derivation_satellite import gradient_1satellite, gradient_4satellite

class GradB:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradb'
        self.incompressible = incompressible
    
    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        inc = 'I' * self.incompressible
        if traj:
            for axisb in ('x', 'y', 'z'):
                if traj_param['nbsatellite'] == 1:
                    gradb = gradient_1satellite(
                        np.array([dic_quant[f"{inc}b{axisb}"]]),
                        traj_param
                    )
                    for i,axisd in enumerate(('x', 'y', 'z')):
                        ds_name = f"{inc}d{axisd}b{axisb}"
                        file.create_dataset(
                            ds_name,
                            data=gradb[i]
                        )
                elif traj_param['nbsatellite'] == 4:
                    gradb = gradient_4satellite(
                        dic_quant,
                        f"{inc}b{axisb}",
                        traj_param,
                    )
                    for i,axisd in enumerate(('x', 'y', 'z')):
                        ds_name = f"{inc}d{axisd}b{axisb}"
                        file.create_dataset(
                            ds_name,
                            data=gradb[i]
                        )
        else:
            for axisb in ('x', 'y', 'z'):
                dxb, dyb, dzb = derivation.grad(
                    dic_quant[f"{inc}b{axisb}"], 
                    dic_param["c"], 
                    precision = 4, 
                    period = True
                )
                for axisd in ('x', 'y', 'z'):
                    ds_name = f"{inc}d{axisd}b{axisb}"
                    file.create_dataset(
                        ds_name,
                        data = eval(f"d{axisd}b"),
                        shape = dic_param["N"],
                        dtype = np.float64,
                    )

def load(incompressible=False):
    gradb = GradB(incompressible=incompressible)
    return gradb.name, gradb