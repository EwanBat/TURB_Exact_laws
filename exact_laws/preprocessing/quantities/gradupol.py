import numpy as np
import numexpr as ne

from ...mathematical_tools import derivation

class GradUPol:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'gradupol'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            raise NotImplementedError("")
        
        if not "gamma" in dic_param.keys():
            dic_param['gamma'] = 5/3
            gamma = 5/3
        else: 
            gamma = dic_param['gamma']
        if "cst" in dic_param.keys():
            cst = dic_param['cst']
        else: 
            cst = (np.mean(ne.evaluate(f"(ppar+pperp+pperp)/3", local_dict=dic_quant))
                    / np.mean(ne.evaluate(f"rho**(gamma)", local_dict=dic_quant, global_dict=dic_param)))
        rho = dic_quant['rho']
        
        if gamma != 1:
            upol = ne.evaluate("cst/(gamma-1)*rho**(gamma-1)")
        else: 
            upol = ne.evaluate("cst*log(rho)")
            
        dxupol, dyupol, dzupol = derivation.grad(
            upol, 
            dic_param["c"], 
            precision = 4, 
            period = True
        )
        for axisd in ('x', 'y', 'z'):
            ds_name = f"d{axisd}upol"
            file.create_dataset(
                ds_name,
                data = eval(f"d{axisd}upol"),
                shape = dic_param["N"],
                dtype = np.float64,
            )      
        
def load(incompressible=False):
    gradupol = GradUPol(incompressible=incompressible)
    return gradupol.name, gradupol
