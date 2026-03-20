import numpy as np
import numexpr as ne

from .pcgl import get_original_quantity

class UCgl:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'ucgl'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            raise NotImplementedError("")
        
        if not ("pparcgl" in dic_quant.keys() or "pperpcgl" in dic_quant.keys()):
            get_original_quantity(dic_quant, dic_param)
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = ne.evaluate("(pparcgl+pperpcgl+pperpcgl)/2/rho", local_dict=dic_quant),
            shape = dic_param["N"],
            dtype = np.float64,
        )
        del(dic_quant['pparcgl'],dic_quant['pperpcgl'])

def load(incompressible=False):
    ucgl = UCgl(incompressible=incompressible)
    return ucgl.name, ucgl
