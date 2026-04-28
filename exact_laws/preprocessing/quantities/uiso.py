import numpy as np
import numexpr as ne


class UIso:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'uiso'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        if self.incompressible:
            raise NotImplementedError("")
        
        ds_name = f"{self.name}"
        if traj:
            file.create_dataset(
                ds_name,
                data=ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant)
            )
        else:
            file.create_dataset(
                ds_name,
                data = ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant),
                shape = dic_param["N"],
                dtype = np.float64,
            )

def load(incompressible=False):
    uiso = UIso(incompressible=incompressible)
    return uiso.name, uiso
