import numpy as np


class Rho:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'rho'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, ltraj_list: list = None, nbsatellites: int = None):
        ds_name = f"{self.name}"
        if self.incompressible:
            file.create_dataset(
                ds_name,
                data=np.ones(dic_param["N"]),
                shape=dic_param["N"],
                dtype=np.float64,
            )
        else:
            file.create_dataset(
                ds_name,
                data=dic_quant[ds_name],
                shape=dic_param["N"],
                dtype=np.float64,
            )


def load(incompressible=False):
    rho = Rho(incompressible=incompressible)
    return rho.name, rho
