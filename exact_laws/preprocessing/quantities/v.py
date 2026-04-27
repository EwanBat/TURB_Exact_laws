import numpy as np


class V:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'v'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, ltraj_list: list = None, nbsatellites: int = None):
        if self.incompressible:
            #doit modifier dic_quant
            raise NotImplementedError("")
        for axis in ('x', 'y', 'z'):
            ds_name = f"{self.name}{axis}"
            file.create_dataset(
                ds_name,
                data = dic_quant[ds_name],
                shape = dic_param["N"],
                dtype = np.float64,
            )


def load(incompressible=False):
    v = V(incompressible=incompressible)
    return v.name, v
