import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

class V2:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'v2'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        inc = 'I' * self.incompressible
        v2 = dic_quant[f"{inc}vx"]*dic_quant[f"{inc}vx"] + dic_quant[f"{inc}vy"]*dic_quant[f"{inc}vy"] + dic_quant[f"{inc}vz"]*dic_quant[f"{inc}vz"]
        
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = v2,
            shape = dic_param["N"],
            dtype = np.float64,
        )    
        
def load(incompressible=False):
    v2 = V2(incompressible=incompressible)
    return v2.name, v2
