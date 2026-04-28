import numpy as np
import numexpr as ne


class PIso:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'piso'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        if traj:
            if self.incompressible:
                ds_name = f"{self.name}"
                file.create_dataset(
                    ds_name,
                    data=ne.evaluate(f"(ppar+pperp+pperp)/3", local_dict=dic_quant)
                )
            else:
                ds_name = f"{self.name}"
                file.create_dataset(
                    ds_name,
                    data=ne.evaluate(f"(ppar+pperp+pperp)/3/rho", local_dict=dic_quant)
                )
        else:
            if self.incompressible:
                ds_name = f"{self.name}"
                file.create_dataset(
                    ds_name,
                    data = ne.evaluate(f"(ppar+pperp+pperp)/3", local_dict=dic_quant),
                    shape = dic_param["N"],
                    dtype = np.float64,
                )
            else:
                ds_name = f"{self.name}"
                file.create_dataset(
                    ds_name,
                    data = ne.evaluate(f"(ppar+pperp+pperp)/3/rho", local_dict=dic_quant),
                    shape = dic_param["N"],
                    dtype = np.float64,
                )

def load(incompressible=False):
    piso = PIso(incompressible=incompressible)
    return piso.name, piso
