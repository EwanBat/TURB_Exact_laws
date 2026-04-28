import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation
from .j import get_original_quantity
from trajectories.derivation_satellite import divergence_1satellite

class DivJ:
    def __init__(self, incompressible=False):
        self.name = "I" * incompressible + "divj"
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        if traj:
            if self.incompressible:
                if traj_param.get('nbsatellites') == 1:
                    divj = divergence_1satellite(
                        np.array([dic_quant[f"jx"], dic_quant[f"jy"], dic_quant[f"jz"]]),
                        traj_param
                    )
            else:
                if not ("jcx" in dic_quant.keys() or "jcy" in dic_quant.keys() or "jcz" in dic_quant.keys()):
                    get_original_quantity(dic_quant, dic_param, inc=False)
                if traj_param.get('nbsatellites') == 1:
                    divj = divergence_1satellite(
                        np.array([dic_quant[f"jcx"], dic_quant[f"jcy"], dic_quant[f"jcz"]]),
                        traj_param
                    )
        else:
            if self.incompressible:
                if not ("jx" in dic_quant.keys() or "jy" in dic_quant.keys() or "jz" in dic_quant.keys()):
                    get_original_quantity(dic_quant, dic_param)
                divj = derivation.div(
                    [dic_quant['jx'], dic_quant['jy'], dic_quant['jz']],
                    dic_param["c"],
                    precision = 4,
                    period = True,
                )
                
            else:
                if not ("jcx" in dic_quant.keys() or "jcy" in dic_quant.keys() or "jcz" in dic_quant.keys()):
                    get_original_quantity(dic_quant, dic_param, inc=False)
                divj = derivation.div(
                    [dic_quant['jcx'], dic_quant['jcy'], dic_quant['jcz']],
                    dic_param["c"],
                    precision = 4,
                    period = True,
                )
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = divj,
            shape = dic_param["N"],
            dtype = np.float64,
        )

def load(incompressible=False):
    divj = DivJ(incompressible=incompressible)
    return divj.name, divj
