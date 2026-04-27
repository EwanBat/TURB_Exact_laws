import numpy as np
import numexpr as ne


class UGyr:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'ugyr'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, ltraj_list: list = None, nbsatellites: int = None):
        if self.incompressible:
            raise NotImplementedError("")
        
        ds_name = f"{self.name}"
        file.create_dataset(
            ds_name,
            data = ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant),
            shape = dic_param["N"],
            dtype = np.float64,
        )

def load(incompressible=False):
    ugyr = UGyr(incompressible=incompressible)
    return ugyr.name, ugyr
