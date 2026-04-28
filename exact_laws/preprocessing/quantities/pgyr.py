import numpy as np
import numexpr as ne


class PGyr:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'pgyr'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param, traj: bool = False, traj_param: dict = None):
        if traj:
            if self.incompressible:
                for axis in ('par', 'perp'):
                    ds_name = f"{self.name[:-3]}{axis}"
                    file.create_dataset(ds_name, data=dic_quant[ds_name[1:]])
            else:
                for axis in ('par', 'perp'):
                    ds_name = f"{self.name[:-3]}{axis}"
                    file.create_dataset(
                        ds_name,
                        data=ne.evaluate(f"{ds_name}/rho", local_dict=dic_quant)
                    )
        else:
            if self.incompressible:
                for axis in ('par', 'perp'):
                    ds_name = f"{self.name[:-3]}{axis}"
                    file.create_dataset(
                        ds_name,
                        data = dic_quant[ds_name[1:]],
                        shape = dic_param["N"],
                        dtype = np.float64,
                    )
            else:
                for axis in ('par', 'perp'):
                    ds_name = f"{self.name[:-3]}{axis}"
                    file.create_dataset(
                        ds_name,
                        data = ne.evaluate(f"{ds_name}/rho", local_dict=dic_quant),
                        shape = dic_param["N"],
                        dtype = np.float64,
                    )

def load(incompressible=False):
    pgyr = PGyr(incompressible=incompressible)
    return pgyr.name, pgyr
