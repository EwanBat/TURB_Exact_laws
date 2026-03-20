import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

class Vnorm:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'vnorm'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        inc = 'I' * self.incompressible
        v = np.sqrt(dic_quant[f"{inc}vx"]*dic_quant[f"{inc}vx"] + dic_quant[f"{inc}vy"]*dic_quant[f"{inc}vy"] + dic_quant[f"{inc}vz"]*dic_quant[f"{inc}vz"])
        
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = v,
            shape = dic_param["N"],
            dtype = np.float64,
        )    
        
def load(incompressible=False):
    v = Vnorm(incompressible=incompressible)
    return v.name, v
