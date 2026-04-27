import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from trajectories.derivation_satellite import divergence_1satellite

class DivV:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'divv'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, ltraj_list: list = None, nbsatellites: int = None):
        inc = 'I' * self.incompressible
        if traj:
            if self.incompressible:
                if nbsatellites == 1:
                    divv = divergence_1satellite(
                        np.array([dic_quant[f"vx"], dic_quant[f"vy"], dic_quant[f"vz"]]),
                        ltraj_list
                    )
            else:
                if nbsatellites == 1:
                    divv = divergence_1satellite(
                        np.array([dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]]),
                        ltraj_list
                    )
        else:
            divv = derivation.div(
                [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]], 
                dic_param["c"], 
                precision = 4, 
                period = True
            )
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data = divv,
                shape = dic_param["N"],
                dtype = np.float64,
            )    
        
def load(incompressible=False):
    divv = DivV(incompressible=incompressible)
    return divv.name, divv
