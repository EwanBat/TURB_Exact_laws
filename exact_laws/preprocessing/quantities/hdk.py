import numpy as np

from ...mathematical_tools.derivation import laplacien

class HdK:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'hdk'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        
        for axisd in ('x', 'y', 'z'):
            hdk = dic_param["nu"] * -laplacien(
                -laplacien(
                    -laplacien(
                        -laplacien(dic_quant[f"v{axisd}"],
                                [1,1,dic_param["an_hd"]],dic_param["c"])
                        ,[1,1,dic_param["an_hd"]],dic_param["c"])
                    ,[1,1,dic_param["an_hd"]],dic_param["c"])
                ,[1,1,dic_param["an_hd"]],dic_param["c"])
        
            ds_name = f"{self.name}{axisd}"
            file.create_dataset(
                ds_name,
                data = hdk,
                shape = dic_param["N"],
                dtype = np.float64,
            )    
        
def load(incompressible=False):
    hdk = HdK(incompressible=incompressible)
    return hdk.name, hdk
