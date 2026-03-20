import numpy as np
import numexpr as ne
from .b import get_original_quantity

from ...mathematical_tools import derivation


class DivB:
    def __init__(self, incompressible=False):
        self.name = "I" * incompressible + "divb"
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            divb = derivation.div(
                [dic_quant[f"bx"], dic_quant[f"by"], dic_quant[f"bz"]], 
                dic_param["c"], 
                precision = 4, 
                period = True
            )
        else:
            if not ("vax" in dic_quant.keys() or "vay" in dic_quant.keys() or "vaz" in dic_quant.keys()):
                get_original_quantity(dic_quant, dic_param)
            divb = derivation.div(
                [dic_quant[f"vax"], dic_quant[f"vay"], dic_quant[f"vaz"]],
                dic_param["c"],
                precision = 4,
                period = True,
            )
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = divb,
            shape = dic_param["N"],
            dtype = np.float64,
        )

def load(incompressible=False):
    divb = DivB(incompressible=incompressible)
    return divb.name, divb
