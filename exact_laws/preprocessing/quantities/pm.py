import numpy as np
import numexpr as ne
from .b import get_original_quantity

class PM:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'pm'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data = ne.evaluate("bx*bx/2+by*by/2+bz*bz/2", local_dict=dic_quant),
                shape = dic_param["N"],
                dtype = np.float64,
            )
        else:
            if not ("vax" in dic_quant.keys() or "vay" in dic_quant.keys() or "vaz" in dic_quant.keys()):
                get_original_quantity(dic_quant, dic_param)
            ds_name = f"{self.name}"
            file.create_dataset(
                ds_name,
                data = ne.evaluate("(vax*vax+vay*vay+vaz*vaz)/2", local_dict=dic_quant),
                shape = dic_param["N"],
                dtype = np.float64,
            )

def load(incompressible=False):
    pm = PM(incompressible=incompressible)
    return pm.name, pm
