import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from trajectories.derivation_satellite import gradient_1satellite

class GradRho:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradrho'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        if traj:
            if self.incompressible:
                # For incompressible flows, rho=1 so grad(rho)=0
                rho_shape = dic_quant['rho'].shape if 'rho' in dic_quant else dic_quant[next(iter(dic_quant.keys()))].shape
                for i,axisd in enumerate(('x', 'y', 'z')):
                    ds_name = f"Igradrho{axisd}"
                    file.create_dataset(
                        ds_name,
                        data=np.zeros(rho_shape)
                    )
            else:
                if traj_param.get('nbsatellites') == 1:
                    gradrho = gradient_1satellite(
                        np.array([dic_quant[f"rho"]]),
                        traj_param
                    )
                    for i,axisd in enumerate(('x', 'y', 'z')):
                        ds_name = f"gradrho{axisd}"
                        file.create_dataset(
                            ds_name,
                            data=gradrho[i]
                        )
        else:
            if self.incompressible:
                for axisd in ('x', 'y', 'z'):
                    ds_name = f"Id{axisd}rho"
                    file.create_dataset(
                        ds_name,
                        data = np.zeros(dic_param["N"]),
                        shape = dic_param["N"],
                        dtype = np.float64,
                    ) 
            
            else:
                dxrho, dyrho, dzrho = derivation.grad(
                    dic_quant[f"rho"], 
                    dic_param["c"], 
                    precision = 4, 
                    period = True
                )
                for axisd in ('x', 'y', 'z'):
                    ds_name = f"d{axisd}rho"
                    file.create_dataset(
                        ds_name,
                        data = eval(f"d{axisd}rho"),
                        shape = dic_param["N"],
                        dtype = np.float64,
                    )      
        
def load(incompressible=False):
    gradrho = GradRho(incompressible=incompressible)
    return gradrho.name, gradrho
