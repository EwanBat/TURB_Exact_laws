import numpy as np
import numexpr as ne

def get_original_quantity(dic_quant, dic_param, delete=False):
    for axis in ('x', 'y', 'z'):
        dic_quant[f'va{axis}'] = ne.evaluate(f"b{axis}/sqrt(rho)", local_dict=dic_quant)
        if delete: 
            del(dic_quant[f'b{axis}'])
    #if delete: 
    #    del(dic_quant['rho'])

class B:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'b'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
        if self.incompressible:
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = dic_quant[ds_name[1:]],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )
        else:
            if not ("vax" in dic_quant.keys() or "vay" in dic_quant.keys() or "vaz" in dic_quant.keys()):
                get_original_quantity(dic_quant, dic_param)
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = dic_quant['va'+axis],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )


def load(incompressible=False):
    b = B(incompressible=incompressible)
    return b.name, b
