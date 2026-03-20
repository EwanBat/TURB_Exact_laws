import numpy as np
import numexpr as ne

from .b import get_original_quantity


class Bnorm:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'bnorm'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            b = np.sqrt(dic_quant[f"bx"]*dic_quant[f"bx"] + 
                        dic_quant[f"by"]*dic_quant[f"by"] + 
                        dic_quant[f"bz"]*dic_quant[f"bz"])
        
        else:
            if not ("vax" in dic_quant.keys() or "vay" in dic_quant.keys() or "vaz" in dic_quant.keys()):
                get_original_quantity(dic_quant, dic_param)
            b = np.sqrt(dic_quant[f"vax"]*dic_quant[f"vax"] + 
                        dic_quant[f"vay"]*dic_quant[f"vay"] + 
                        dic_quant[f"vaz"]*dic_quant[f"vaz"])
            
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = b,
            shape = dic_param["N"],
            dtype = np.float64,
        )    
        
def load(incompressible=False):
    b = Bnorm(incompressible=incompressible)
    return b.name, b
