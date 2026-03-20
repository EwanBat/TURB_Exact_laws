import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

class GradV2:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradv2'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        inc = 'I' * self.incompressible
        v2 = dic_quant[f"{inc}vx"]*dic_quant[f"{inc}vx"] + dic_quant[f"{inc}vy"]*dic_quant[f"{inc}vy"] + dic_quant[f"{inc}vz"]*dic_quant[f"{inc}vz"]
        dxv2, dyv2, dzv2 = derivation.grad(v2, 
            dic_param["c"], 
            precision = 4, 
            period = True
        )
        for axisd in ('x', 'y', 'z'):
            ds_name = f"{inc}d{axisd}v2"
            file.create_dataset(
                ds_name,
                data = eval(f"d{axisd}v2"),
                shape = dic_param["N"],
                dtype = np.float64,
            )      
        
def load(incompressible=False):
    gradv2 = GradV2(incompressible=incompressible)
    return gradv2.name, gradv2
