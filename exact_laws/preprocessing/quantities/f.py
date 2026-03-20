import numpy as np

from ...mathematical_tools.derivation import cdiff

class F:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'f'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        
        if not "a1_forc" in dic_param.keys():
            dic_param['a1_forc'] = 0.5
            a1 = 0.5
        else: 
            a1 = dic_param['a1_forc']
                
        f_langevin = a1 * dic_quant["fp"] + (1-a1) * dic_quant["fm"]
        
        ds_name = f"{self.name}x"
        file.create_dataset(
            ds_name,
            data = cdiff(f_langevin, dic_param['c'][1], 1),
            shape = dic_param["N"],
            dtype = np.float64,
        ) 
        
        ds_name = f"{self.name}y"
        file.create_dataset(
            ds_name,
            data = - cdiff(f_langevin, dic_param['c'][0], 0),
            shape = dic_param["N"],
            dtype = np.float64,
        ) 
        
        ds_name = f"{self.name}z"
        file.create_dataset(
            ds_name,
            data = np.zeros_like(f_langevin),
            shape = dic_param["N"],
            dtype = np.float64,
        )       
        
def load(incompressible=False):
    f = F(incompressible=incompressible)
    return f.name, f
