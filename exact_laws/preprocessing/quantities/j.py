import numpy as np

from ...mathematical_tools import derivation

def get_original_quantity(dic_quant, dic_param, delete=False, inc=True):
    dic_quant['jx'], dic_quant['jy'], dic_quant['jz'] = derivation.rot(
        [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]],
        dic_param["c"],
        precision = 4,
        period = True,
    )
    if delete: 
        del(dic_quant['bx'],dic_quant['by'],dic_quant['bz'])
    
    if not inc:
        dic_quant['jcx'] = dic_quant['jx']/dic_quant['rho']
        del(dic_quant['jx'])
        dic_quant['jcy'] = dic_quant['jy']/dic_quant['rho']
        del(dic_quant['jy'])
        dic_quant['jcz'] = dic_quant['jz']/dic_quant['rho']
        del(dic_quant['jz'])

class J:
    def __init__(self, incompressible=False):
        self.name = 'I' * incompressible + 'j'
        self.incompressible = incompressible

    def create_datasets(self, file, dic_quant, dic_param):
            
        if self.incompressible:
            if not ("jx" in dic_quant.keys() or "jy" in dic_quant.keys() or "jz" in dic_quant.keys()):
                get_original_quantity(dic_quant, dic_param)
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = dic_quant[ds_name[1:]],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )
        else:
            if not ("jcx" in dic_quant.keys() or "jcy" in dic_quant.keys() or "jcz" in dic_quant.keys()):
                get_original_quantity(dic_quant, dic_param, inc=False)
            for axis in ('x', 'y', 'z'):
                ds_name = f"{self.name}{axis}"
                file.create_dataset(
                    ds_name,
                    data = dic_quant[f'jc{axis}'],
                    shape = dic_param["N"],
                    dtype = np.float64,
                )


def load(incompressible=False):
    j = J(incompressible=incompressible)
    return j.name, j
